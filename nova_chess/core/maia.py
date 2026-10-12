"""
Обёртка над Maia-2 — вторым человеческим движком.

Maia-2 отличается от Nova двумя важными вещами:

* она принимает **рейтинг обоих игроков** (своего и соперника), а не только свой;
* она обучена **отдельно по рейтинговым диапазонам**, поэтому разница
  между 1100 и 1900 выражена заметно сильнее, чем у Nova.

Диапазон Maia-2 — примерно 1100–1900. Значения вне его обрезаются,
поэтому она не подходит для имитации гроссмейстера или новичка-перворазрядника.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import chess

# Maia-2 обучена на этих границах; вне диапазона отвечает неадекватно
MAIA_MIN_RATING = 1100
MAIA_MAX_RATING = 1900

# Скорость игры, под которую обучена модель
MAIA_SPEED = "rapid"


@dataclass(frozen=True)
class MaiaMove:
    """Ход с вероятностью по мнению Maia."""

    uci: str
    probability: float

    @property
    def percent(self) -> float:
        """Вероятность в процентах."""
        return self.probability * 100.0


class MaiaEngine:
    """Человеческий движок Maia-2.

    Пример:
        >>> engine = MaiaEngine()
        >>> engine.predict(chess.Board(), elo_self=1500, elo_oppo=1500)[:3]
    """

    def __init__(self, speed: str = MAIA_SPEED) -> None:
        """Загружает веса Maia-2.

        Args:
            speed: 'rapid' или 'blitz' — модель под темп игры.

        Raises:
            ImportError: если пакет maia2 не установлен.
        """
        try:
            from maia2 import inference, model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Maia-2 не установлена. Выполните: pip install maia2"
            ) from exc

        self._inference = inference
        self._model = model.from_pretrained(type=speed, device="cpu")
        self._prepared = inference.prepare()
        self._lock = threading.Lock()
        self.speed = speed

    def clamp_rating(self, rating: int) -> int:
        """Приводит рейтинг к диапазону, на котором обучена Maia."""
        return max(MAIA_MIN_RATING, min(MAIA_MAX_RATING, int(rating)))

    def is_in_range(self, rating: int) -> bool:
        """Попадает ли рейтинг в обученный диапазон."""
        return MAIA_MIN_RATING <= rating <= MAIA_MAX_RATING

    def raw_prediction(
        self, board: chess.Board, elo_self: int, elo_oppo: int
    ) -> tuple[dict[str, float], float]:
        """Сырые вероятности ходов и ожидаемый результат партии.

        Returns:
            Кортеж (вероятности ходов, ожидаемый счёт 0..1).
        """
        elo_self = self.clamp_rating(elo_self)
        elo_oppo = self.clamp_rating(elo_oppo)

        with self._lock:
            move_probs, win_prob = self._inference.inference_each(
                self._model,
                self._prepared,
                board.fen(),
                elo_self,
                elo_oppo,
            )
        return move_probs, win_prob

    def move_probabilities(
        self,
        board: chess.Board,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
    ) -> dict[str, float]:
        """Вероятности всех легальных ходов по мнению Maia.

        Возвращает только легальные ходы в текущей позиции.
        """
        probs, _ = self.raw_prediction(board, elo_self, elo_oppo)
        legal = {m.uci() for m in board.legal_moves}

        filtered = {uci: p for uci, p in probs.items() if uci in legal}
        total = sum(filtered.values())
        if total <= 0:
            return {}

        return {uci: p / total for uci, p in filtered.items()}

    def predict(
        self,
        board: chess.Board,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        *,
        top: int = 5,
    ) -> list[MaiaMove]:
        """Топ-N ходов, которые сделал бы человек рейтинга elo_self."""
        probs = self.move_probabilities(board, elo_self, elo_oppo)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return [MaiaMove(uci, p) for uci, p in ordered[:top]]

    def expected_score(
        self, board: chess.Board, elo_self: int = 1500, elo_oppo: int = 1500
    ) -> float:
        """Ожидаемый результат партии с точки зрения elo_self (0..1)."""
        _, win_prob = self.raw_prediction(board, elo_self, elo_oppo)
        return float(win_prob)

    def sample_move(
        self,
        board: chess.Board,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        *,
        sharpness: float = 1.0,
        rng=None,
    ) -> chess.Move | None:
        """Выбирает ход, семплируя распределение Maia."""
        import numpy as np

        probs = self.move_probabilities(board, elo_self, elo_oppo)
        if not probs:
            return None

        rng = rng or np.random.default_rng()
        ucis = list(probs.keys())
        weights = np.array([probs[u] for u in ucis], dtype=np.float64)

        if sharpness > 1.0:
            weights = weights ** sharpness
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(ucis)) / len(ucis)
        else:
            weights /= total

        return chess.Move.from_uci(ucis[int(rng.choice(len(ucis), p=weights))])
