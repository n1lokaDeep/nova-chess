"""
Движок Nova: загрузка модели и предсказание человеческих ходов.

Nova — policy-only сеть: один проход, без дерева поиска.
Она отвечает на вопрос «какой ход сделал бы человек рейтинга R»,
а не «какой ход объективно лучший».
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from .encoding import fen_to_planes, make_conditioning

# Пространство ходов Nova: 64 * 64 откуда-куда + 3 блока превращений
BOARD_SQUARES = 64
PROMOTION_BLOCK = 4096
NUM_MOVE_INDICES = 16384

# Смещения для превращений
PROMO_OFFSETS = {
    chess.KNIGHT: PROMOTION_BLOCK,
    chess.BISHOP: PROMOTION_BLOCK * 2,
    chess.ROOK: PROMOTION_BLOCK * 3,
}


@dataclass(frozen=True)
class MoveCandidate:
    """Один возможный ход с оценкой вероятности."""

    uci: str
    probability: float

    @property
    def percent(self) -> float:
        """Вероятность в процентах."""
        return self.probability * 100.0


def _move_to_index(move: chess.Move) -> int:
    """Переводит ход python-chess в индекс пространства Nova."""
    index = move.from_square * BOARD_SQUARES + move.to_square
    if move.promotion is not None:
        index += PROMO_OFFSETS.get(move.promotion, 0)
    return index


def index_to_uci(index: int, board: chess.Board | None = None) -> str:
    """Переводит индекс Nova в строку UCI.

    Превращение в ферзя в пространстве Nova имеет смещение 0 и неотличимо
    от обычного хода по индексу. Если передан `board`, буква превращения
    восстанавливается по позиции; иначе для такого индекса возвращается
    обычный ход без суффикса.

    Args:
        index: индекс хода в пространстве Nova.
        board: позиция, из которой сделан ход (для восстановления превращения).

    Returns:
        Строка в формате UCI.
    """
    promo = ""
    raw = int(index)
    if raw >= PROMOTION_BLOCK * 3:
        promo, raw = "r", raw - PROMOTION_BLOCK * 3
    elif raw >= PROMOTION_BLOCK * 2:
        promo, raw = "b", raw - PROMOTION_BLOCK * 2
    elif raw >= PROMOTION_BLOCK:
        promo, raw = "n", raw - PROMOTION_BLOCK

    uci = chess.square_name(raw // BOARD_SQUARES) + chess.square_name(raw % BOARD_SQUARES)

    if promo:
        return uci + promo

    # Смещение 0: либо обычный ход, либо превращение в ферзя.
    # Различить можно только по позиции.
    if board is not None:
        candidate = chess.Move.from_uci(uci + "q")
        if candidate in board.legal_moves:
            return uci + "q"

    return uci


class NovaEngine:
    """Обёртка над ONNX-моделью Nova.

    Потокобезопасна: сессия ONNX защищена блокировкой.

    Пример:
        >>> engine = NovaEngine("nova_v3b.onnx")
        >>> engine.predict(chess.Board(), rating=1500)[0]
        MoveCandidate(uci='e2e4', probability=0.55)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        intra_op_threads: int = 6,
    ) -> None:
        """Загружает модель.

        Args:
            model_path: путь к `nova_v3b.onnx`. Рядом должен лежать
                файл весов `nova_v3b.onnx.data`.
            intra_op_threads: число потоков ONNX внутри одной операции.

        Raises:
            FileNotFoundError: если модель или файл весов отсутствуют.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        data_file = model_path.with_suffix(model_path.suffix + ".data")
        if not data_file.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {data_file}\n"
                "Nova хранит веса отдельно — положите .onnx и .onnx.data рядом."
            )

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = intra_op_threads

        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()
        self.model_path = model_path

    def raw_logits(self, board: chess.Board, rating: int,
                   classical: float, aggression: float) -> np.ndarray:
        """Возвращает сырые логиты модели по всем 16384 индексам."""
        planes = fen_to_planes(board.fen())[np.newaxis]
        conditioning = make_conditioning(rating, classical, aggression)

        with self._lock:
            logits = self._session.run(
                None, {"positions": planes, "conditioning": conditioning}
            )[0][0]
        return logits

    def move_probabilities(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
    ) -> dict[str, float]:
        """Вероятности всех легальных ходов по мнению Nova.

        Args:
            board: текущая позиция.
            rating: рейтинг игрока, 800–2700.
            classical: ось стиля 0..1.
            aggression: ось агрессии 0..1.

        Returns:
            Словарь {uci: вероятность}, сумма равна 1.
        """
        logits = self.raw_logits(board, rating, classical, aggression)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        indices = np.fromiter(
            (_move_to_index(m) for m in legal_moves),
            dtype=np.int64, count=len(legal_moves),
        )
        values = logits[indices].astype(np.float64)

        # Softmax только по легальным ходам: нелегальные не должны влиять
        values -= values.max()
        exp_values = np.exp(values)
        total = exp_values.sum()
        if total <= 0:
            return {}
        probabilities = exp_values / total

        return {
            move.uci(): float(probabilities[i])
            for i, move in enumerate(legal_moves)
        }

    def predict(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        top: int = 5,
    ) -> list[MoveCandidate]:
        """Топ-N ходов, которые сделал бы человек заданного рейтинга.

        Returns:
            Список MoveCandidate, отсортированный по убыванию вероятности.
        """
        probabilities = self.move_probabilities(board, rating, classical, aggression)
        ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return [MoveCandidate(uci, p) for uci, p in ordered[:top]]

    def sample_move(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        sharpness: float = 3.0,
        min_relative_prob: float = 0.12,
        rng: np.random.Generator | None = None,
    ) -> chess.Move | None:
        """Выбирает ход, семплируя из «человеческого» распределения.

        Ключевое отличие от наивного семплирования: хвост распределения
        отбрасывается. Без этого Nova изредка выдаёт ходы, которые человек
        не сделал бы никогда (например, b2b3 на первом ходу с вероятностью 0.1%).

        Args:
            sharpness: >1 обостряет распределение, приближая к топ-ходу.
            min_relative_prob: минимальная доля от вероятности лидера,
                при которой ход ещё рассматривается.
            rng: генератор случайных чисел (для воспроизводимости).

        Returns:
            Выбранный ход или None, если ходов нет.
        """
        probabilities = self.move_probabilities(board, rating, classical, aggression)
        if not probabilities:
            return None

        rng = rng or np.random.default_rng()

        leader = max(probabilities.values())
        candidates = [
            (uci, p) for uci, p in probabilities.items()
            if p >= leader * min_relative_prob
        ]
        if not candidates:
            best = max(probabilities.items(), key=lambda kv: kv[1])
            return chess.Move.from_uci(best[0])

        ucis = [c[0] for c in candidates]
        weights = np.array([c[1] for c in candidates], dtype=np.float64)
        weights = weights ** sharpness
        weights /= weights.sum()

        chosen = ucis[int(rng.choice(len(ucis), p=weights))]
        return chess.Move.from_uci(chosen)
