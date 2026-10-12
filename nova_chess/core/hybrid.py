"""
Nova Chess: движок, который играет как человек, но не зевает.

Объединяет две модели:
  * Nova     — предсказывает ходы, которые делают люди заданного рейтинга
  * Stockfish — проверяет качество и отсеивает откровенные промахи

Баланс между «человечностью» и «силой» задаётся параметром `strictness`.
"""

from __future__ import annotations

import math
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
        strictness: float | None = None,
        pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> list[EvaluatedMove]:
        """Ходы Nova, оценённые Stockfish и отфильтрованные по качеству.

        Args:
            strictness: если None — вычисляется из рейтинга через
                `rating_to_strictness`. Явное значение переопределяет.

        Returns:
            Список EvaluatedMove. Если фильтр недоступен — возвращает
            кандидатов Nova без оценки (loss = 0).
        """
        from .filter import rating_to_strictness

        candidates = [
            (c.uci, c.probability)
            for c in self.nova.predict(board, rating, classical, aggression, top=pool)
        ]
        if not candidates:
            return []

        if self._filter is None:
            return [EvaluatedMove(uci, p, 0, 0) for uci, p in candidates]

        if strictness is None:
            strictness = rating_to_strictness(rating)

        return self._filter.filter(board, candidates, strictness=strictness)

    def choose_move(
        self,
        board: chess.Board,
        rating: int = 1500,
        classical: float = 0.5,
        aggression: float = 0.5,
        *,
        strictness: float | None = None,
        sharpness: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> chess.Move | None:
        """Выбирает ход: Nova предлагает, Stockfish фильтрует.

        Args:
            strictness: None — порог берётся из рейтинга (рекомендуется).
                Явное значение 0 отключает фильтр.
            sharpness: >1 концентрирует выбор на вероятном ходе.

        Returns:
            Выбранный ход или None при отсутствии ходов.
        """
        from .filter import rating_to_strictness

        board = board.copy()

        if strictness is None:
            strictness = rating_to_strictness(rating) if self._filter else 0.0

        if self._filter is None or strictness <= 0.01:
            return self.nova.sample_move(
                board, rating, classical, aggression,
                sharpness=max(1.0, sharpness), rng=rng
            )

        evaluated = self.analyzed_moves(
            board, rating, classical, aggression, strictness=strictness
        )
        if not evaluated:
            return None

        rng = rng or np.random.default_rng()

        # Если фильтр оставил один ход — берём его без случайности
        if len(evaluated) == 1:
            return chess.Move.from_uci(evaluated[0].uci)

        ucis = [e.uci for e in evaluated]
        weights = np.array([e.nova_probability for e in evaluated], dtype=np.float64)
        weights = weights ** max(1.0, sharpness)
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(ucis)) / len(ucis)
        else:
            weights /= total

        chosen = ucis[int(rng.choice(len(ucis), p=weights))]
        return chess.Move.from_uci(chosen)

    def stockfish_moves(self, board: chess.Board, *, top: int = 5,
                        depth: int | None = None) -> list[tuple[str, float]]:
        """Лучшие ходы по Stockfish для сравнения с человеческими.

        Возвращает список (uci, вес), где вес — нормированная «уверенность»
        на основе оценки. Нужен только для наглядного сравнения:
        в отличие от Nova, эти ходы не зависят от рейтинга.

        Args:
            top: сколько ходов вернуть.
            depth: глубина анализа; по умолчанию на 6 больше фильтра.

        Returns:
            Список пар (uci, вероятность), отсортированный по убыванию.
        """
        if self._filter is None:
            raise RuntimeError("Подсказки Stockfish требуют Stockfish")

        from .filter import _score_to_cp

        limit = chess.engine.Limit(depth=depth or self.filter_depth + 6)
        with self._filter._lock:
            raw = self._filter._engine.analyse(board, limit, multipv=top)

        infos = raw if isinstance(raw, list) else [raw]

        # Оценка приводится к перспективе ТОГО, ЧЕЙ ХОД.
        # Критично: если сортировать по перспективе белых, то при ходе
        # чёрных «лучшим» окажется худший для них ход — движок начнёт
        # отдавать фигуры бесплатно, потому что выберет минимум для себя.
        mover = board.turn

        scored: list[tuple[str, int]] = []
        for i in infos:
            if not isinstance(i, dict):
                continue
            pv = i.get("pv") or []
            if not pv or "score" not in i:
                continue
            cp = _score_to_cp(i["score"], mover)
            scored.append((pv[0].uci(), cp))

        if not scored:
            return []

        # Ранжируем по оценке: лучший ход получает вес, многократно
        # превосходящий остальные. Это отражает суть Stockfish —
        # он выбирает ОДИН ход, а не распределение.
        scored.sort(key=lambda x: x[1], reverse=True)
        best_cp = scored[0][1]

        weights: list[float] = []
        for rank, (_, cp) in enumerate(scored):
            if rank == 0:
                weights.append(1.0)
                continue
            # Каждые 50 сп отставания уменьшают вес вчетверо
            penalty = math.exp(-(best_cp - cp) / 50.0)
            # Дополнительный штраф за позицию в списке (Stockfish ранжирует)
            weights.append(penalty * (0.9 ** rank))

        total = sum(weights)
        if total <= 0:
            return [(scored[0][0], 1.0)]

        return [
            (uci, w / total)
            # strict=True: списки обязаны совпадать по длине, иначе
            # молча потеряем ход — лучше сразу увидеть ошибку
            for (uci, _), w in zip(scored, weights, strict=True)
        ]

    def evaluate_position(self, board: chess.Board, depth: int | None = None) -> int:
        """Оценка позиции в сантипешках С ТОЧКИ ЗРЕНИЯ БЕЛЫХ.

        Положительное значение — перевес белых, отрицательное — чёрных.
        Знак НЕ зависит от того, чей сейчас ход, поэтому полоса оценки
        не переворачивается каждый полуход.

        Для графика используйте `evaluate_for_display` — она сжимает
        крайние значения, чтобы полоса не улетала в бесконечность.
        """
        if self._filter is None:
            raise RuntimeError("Оценка позиции требует Stockfish")

        from .filter import _score_to_cp

        with self._filter._lock:
            info = self._filter._engine.analyse(
                board, chess.engine.Limit(depth=depth or self.filter_depth + 4)
            )
        return _score_to_cp(info["score"], chess.WHITE)

    def evaluate_for_display(self, board: chess.Board, depth: int | None = None,
                             *, multipv: int = 3) -> int:
        """Оценка для полосы на графике: сжатая и стабильная.

        ВАЖНО: берётся оценка ЛУЧШЕЙ линии, а не среднее по линиям.
        Усреднять нельзя: вторая и третья линии multipv — это заведомо
        худшие ходы, и их оценка отражает не позицию, а слабость этих
        ходов. Среднее по ним даёт абсурд (выигрышная позиция может
        показать минус).

        Параметр multipv оставлен для совместимости: он влияет только
        на то, сколько линий запрашивается у Stockfish, но результат
        всегда берётся из первой.

        Returns:
            Сантипешки с точки зрения белых, примерно -1500..+1500.
        """
        if self._filter is None:
            raise RuntimeError("Оценка позиции требует Stockfish")

        from .filter import _score_to_cp, to_display_cp

        limit = chess.engine.Limit(depth=depth or self.filter_depth + 10)
        with self._filter._lock:
            raw = self._filter._engine.analyse(board, limit, multipv=1)

        info = raw[0] if isinstance(raw, list) else raw
        if not isinstance(info, dict) or "score" not in info:
            return 0

        cp = _score_to_cp(info["score"], chess.WHITE)
        return to_display_cp(cp)

    def close(self) -> None:
        """Освобождает ресурсы: останавливает Stockfish."""
        if self._filter is not None:
            self._filter.close()

    def __enter__(self) -> HumanLikeEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
