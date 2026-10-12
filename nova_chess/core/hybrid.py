"""
Nova Chess: движок, который играет как человек, но не зевает.

Объединяет две модели:
  * Nova     — предсказывает ходы, которые делают люди заданного рейтинга
  * Stockfish — проверяет качество и отсеивает откровенные промахи

Баланс между «человечностью» и «силой» задаётся параметром `strictness`.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from .engine import MoveCandidate, NovaEngine
from .filter import EvaluatedMove, StockfishFilter

# Сколько лучших ходов Nova передавать в фильтр
DEFAULT_CANDIDATE_POOL = 8


class HumanLikeEngine:
    """Движок, имитирующий человеческую игру с контролем качества.

    Пример:
        >>> engine = HumanLikeEngine("nova_v3b.onnx", "stockfish.exe")
        >>> engine.choose_move(chess.Board(), rating=1500, strictness=0.6)
        'e2e4'
    """

    def __init__(
        self,
        nova_model: str | Path,
        stockfish_path: str | Path | None = None,
        *,
        filter_depth: int = 8,
        intra_op_threads: int = 6,
    ) -> None:
        """Загружает Nova и, опционально, Stockfish.

        Args:
            nova_model: путь к nova_v3b.onnx.
            stockfish_path: путь к Stockfish. Если None — фильтр недоступен,
                движок работает в режиме чистой Nova.
            filter_depth: глубина анализа Stockfish для фильтра.
        """
        self.nova = NovaEngine(nova_model, intra_op_threads=intra_op_threads)
        self._filter: StockfishFilter | None = None
        self.filter_depth = filter_depth

        if stockfish_path is not None:
            self._filter = StockfishFilter(stockfish_path, depth=filter_depth)

    @property
    def has_filter(self) -> bool:
        """Доступен ли фильтр качества."""
        return self._filter is not None

    def human_moves(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        top: int = 5,
    ) -> list[MoveCandidate]:
        """Ходы, которые сделал бы человек — без фильтрации."""
        return self.nova.predict(board, rating, classical, aggression, top=top)

    def analyzed_moves(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        strictness: float = 0.6,
        pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> list[EvaluatedMove]:
        """Ходы Nova, оценённые Stockfish и отфильтрованные по качеству.

        Returns:
            Список EvaluatedMove. Если фильтр недоступен — возвращает
            кандидатов Nova без оценки (loss = 0).
        """
        candidates = [
            (c.uci, c.probability)
            for c in self.nova.predict(board, rating, classical, aggression, top=pool)
        ]
        if not candidates:
            return []

        if self._filter is None:
            return [EvaluatedMove(uci, p, 0, 0) for uci, p in candidates]

        return self._filter.filter(board, candidates, strictness=strictness)

    def choose_move(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        strictness: float = 0.6,
        sharpness: float = 3.0,
        rng: np.random.Generator | None = None,
    ) -> chess.Move | None:
        """Выбирает ход: Nova предлагает, Stockfish фильтрует.

        Args:
            strictness: 0 = чистая Nova, 1 = жёсткий контроль качества.
            sharpness: насколько часто выбирается самый вероятный ход.

        Returns:
            Выбранный ход или None при отсутствии ходов.
        """
        board = board.copy()

        if self._filter is None or strictness <= 0.01:
            return self.nova.sample_move(
                board, rating, classical, aggression, sharpness=sharpness, rng=rng
            )

        evaluated = self.analyzed_moves(
            board, rating, classical, aggression, strictness=strictness
        )
        if not evaluated:
            return None

        rng = rng or np.random.default_rng()
        ucis = [e.uci for e in evaluated]
        weights = np.array([e.nova_probability for e in evaluated], dtype=np.float64)
        weights = weights ** sharpness
        weights /= weights.sum()

        chosen = ucis[int(rng.choice(len(ucis), p=weights))]
        return chess.Move.from_uci(chosen)

    def evaluate_position(self, board: chess.Board, depth: int | None = None) -> int:
        """Оценка позиции в сантипешках (требует Stockfish)."""
        if self._filter is None:
            raise RuntimeError("Оценка позиции требует Stockfish")
        with self._filter._lock:
            info = self._filter._engine.analyse(
                board, chess.engine.Limit(depth=depth or self.filter_depth + 2)
            )
        from .filter import _score_to_cp
        return _score_to_cp(info["score"], board.turn)

    def close(self) -> None:
        """Освобождает ресурсы: останавливает Stockfish."""
        if self._filter is not None:
            self._filter.close()

    def __enter__(self) -> "HumanLikeEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
