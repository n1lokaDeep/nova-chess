"""
Фильтр качества через Stockfish.

Nova не оценивает позиции и не считает варианты — она может предложить ход,
который человек сделал бы, но который объективно плох. Этот модуль
добавляет проверку качества: Stockfish оценивает кандидатов Nova и
отбрасывает те, что теряют слишком много.

Режим фильтрации задаётся параметром `strictness`:
    0.0 — фильтр выключен, чистая Nova (может зевнуть)
    0.5 — умеренный отсев
    1.0 — жёсткий отсев, теряется не более ~0.3 пешки
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

# Заглушка для мата: сантипешки ограничены, мат помечаем большим числом
MATE_SCORE = 10000
MATE_STEP = 10

# Допустимая потеря в сантипешках при strictness = 0 и 1 соответственно
MAX_LOSS_AT_ZERO = 400
MAX_LOSS_AT_ONE = 30


@dataclass(frozen=True)
class EvaluatedMove:
    """Ход с оценкой Stockfish."""

    uci: str
    nova_probability: float
    centipawns: int
    loss: int

    @property
    def is_best(self) -> bool:
        """Совпадает ли ход с лучшим по Stockfish."""
        return self.loss == 0


def _score_to_cp(score: chess.engine.Score, perspective: chess.Color) -> int:
    """Переводит оценку Stockfish в сантипешки с точки зрения стороны."""
    pov = score.pov(perspective)
    if pov.is_mate():
        mate_in = pov.mate()
        # Чем быстрее мат, тем выше оценка
        return MATE_SCORE - abs(mate_in) * MATE_STEP if mate_in > 0 else -MATE_SCORE + abs(mate_in) * MATE_STEP
    return pov.score() or 0


class StockfishFilter:
    """Оценивает ходы через Stockfish и отсеивает слабые.

    Пример:
        >>> filt = StockfishFilter("stockfish.exe", depth=10)
        >>> good = filt.filter(board, candidates, strictness=0.6)
        >>> filt.close()
    """

    def __init__(self, engine_path: str | Path, *, depth: int = 10,
                 threads: int = 4, hash_mb: int = 256) -> None:
        """Запускает процесс Stockfish.

        Args:
            engine_path: путь к исполняемому файлу Stockfish.
            depth: глубина анализа для фильтрации (меньше = быстрее).
            threads: число потоков Stockfish.
            hash_mb: размер хеш-таблицы в мегабайтах.
        """
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Stockfish не найден: {engine_path}")

        self._engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        self._engine.configure({"Threads": threads, "Hash": hash_mb})
        self._lock = threading.Lock()
        self.depth = depth

    def evaluate_move(self, board: chess.Board, uci: str, depth: int | None = None) -> int:
        """Оценка позиции после хода, с точки зрения сделавшего ход.

        Returns:
            Оценка в сантипешках.
        """
        move = chess.Move.from_uci(uci)
        after = board.copy()
        after.push(move)

        with self._lock:
            info = self._engine.analyse(
                after, chess.engine.Limit(depth=depth or self.depth)
            )

        # Оценка даётся с точки зрения того, чей ход ПОСЛЕ —
        # значит для сделавшего ход её надо инвертировать
        return -_score_to_cp(info["score"], after.turn)

    def evaluate_all(
        self, board: chess.Board, candidates: list[tuple[str, float]],
        depth: int | None = None,
    ) -> list[EvaluatedMove]:
        """Оценивает список кандидатов и считает потерю относительно лучшего."""
        scored: list[tuple[str, float, int]] = []
        for uci, probability in candidates:
            try:
                cp = self.evaluate_move(board, uci, depth)
            except Exception:
                continue
            scored.append((uci, probability, cp))

        if not scored:
            return []

        best_cp = max(cp for _, _, cp in scored)
        return [
            EvaluatedMove(uci, prob, cp, best_cp - cp)
            for uci, prob, cp in scored
        ]

    def filter(
        self,
        board: chess.Board,
        candidates: list[tuple[str, float]],
        *,
        strictness: float = 0.6,
        depth: int | None = None,
    ) -> list[EvaluatedMove]:
        """Отсеивает ходы, теряющие слишком много относительно лучшего.

        Args:
            candidates: список (uci, вероятность Nova).
            strictness: 0 = без фильтрации, 1 = жёсткий отсев.

        Returns:
            Отфильтрованный список EvaluatedMove.
            Если после отсева не осталось ходов — возвращает лучший по Stockfish.
        """
        strictness = max(0.0, min(1.0, strictness))
        evaluated = self.evaluate_all(board, candidates, depth)
        if not evaluated:
            return []

        if strictness <= 0.01:
            return evaluated

        allowed_loss = MAX_LOSS_AT_ZERO + (MAX_LOSS_AT_ONE - MAX_LOSS_AT_ZERO) * strictness
        kept = [e for e in evaluated if e.loss <= allowed_loss]

        if kept:
            return kept
        # Всё отсеялось — отдаём объективно лучший
        return [min(evaluated, key=lambda e: e.loss)]

    def close(self) -> None:
        """Останавливает процесс Stockfish."""
        with self._lock:
            try:
                self._engine.quit()
            except Exception:
                pass

    def __enter__(self) -> "StockfishFilter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
