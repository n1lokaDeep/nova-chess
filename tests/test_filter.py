"""Тесты фильтра качества и гибридного движка."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from nova_chess.core.filter import (
    MAX_LOSS_AT_ONE,
    MAX_LOSS_AT_ZERO,
    EvaluatedMove,
    StockfishFilter,
    loss_budget,
    rating_to_strictness,
)
from nova_chess.core.hybrid import HumanLikeEngine

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "nova_v3b.onnx"
STOCKFISH_CANDIDATES = [
    Path(r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"),
    Path(r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-unicode.exe"),
]
STOCKFISH_PATH = next((p for p in STOCKFISH_CANDIDATES if p.exists()), None)

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="нет модели Nova")
needs_stockfish = pytest.mark.skipif(STOCKFISH_PATH is None, reason="нет Stockfish")

# Тактическая позиция: ладья может забрать ферзя
TACTIC_FEN = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"


class TestLossThresholds:
    """Границы допустимой потери."""

    def test_thresholds_ordered(self):
        """Жёсткий отсев строже мягкого."""
        assert MAX_LOSS_AT_ONE < MAX_LOSS_AT_ZERO

    def test_rating_to_strictness_monotonic(self):
        """Строгость растёт вместе с рейтингом."""
        values = [rating_to_strictness(r) for r in range(800, 2701, 100)]
        assert values == sorted(values)

    def test_rating_bounds(self):
        """На краях шкалы строгость равна 0 и максимуму."""
        assert rating_to_strictness(800) == pytest.approx(0.0, abs=0.01)
        assert rating_to_strictness(2700) > 0.8

    def test_rating_clamped(self):
        """Рейтинг вне диапазона обрезается."""
        assert rating_to_strictness(100) == rating_to_strictness(800)
        assert rating_to_strictness(9999) == rating_to_strictness(2700)

    def test_loss_budget_monotonic(self):
        """Бюджет потери падает с ростом строгости."""
        budgets = [loss_budget(s / 10) for s in range(0, 11)]
        assert budgets == sorted(budgets, reverse=True)

    def test_loss_budget_bounds(self):
        """На краях бюджета получаются ожидаемые значения."""
        assert loss_budget(0.0) == MAX_LOSS_AT_ZERO
        assert loss_budget(1.0) == MAX_LOSS_AT_ONE

    def test_low_rating_allows_blunder(self):
        """На 800 допустимая потеря достаточна для зевка фигуры."""
        assert loss_budget(rating_to_strictness(800)) >= 700

    def test_high_rating_is_strict(self):
        """На 2700 допускается только небольшая потеря."""
        assert loss_budget(rating_to_strictness(2700)) < 150

    def test_evaluated_move_is_best(self):
        """loss == 0 означает лучший ход."""
        assert EvaluatedMove("e2e4", 0.5, 100, 0).is_best
        assert not EvaluatedMove("a2a3", 0.1, 50, 50).is_best

    def test_percent_property(self):
        """Вероятность в процентах считается верно."""
        mv = EvaluatedMove("e2e4", 0.5678, 0, 0)
        assert mv.nova_probability * 100 == pytest.approx(56.78)


@needs_stockfish
class TestStockfishFilter:
    """Фильтр на реальном Stockfish."""

    @pytest.fixture(scope="class")
    def filt(self):
        f = StockfishFilter(STOCKFISH_PATH, depth=6)
        yield f
        f.close()

    def test_finds_best_move(self, filt):
        """В тактической позиции лучший ход — взятие ферзя."""
        board = chess.Board(TACTIC_FEN)
        result = filt.evaluate_all(board, [("d1d5", 0.9), ("e1e2", 0.1)])
        assert result
        best = min(result, key=lambda e: e.loss)
        assert best.uci == "d1d5"
        assert best.is_best

    def test_loss_computed(self, filt):
        """Потеря считается относительно лучшего хода."""
        board = chess.Board(TACTIC_FEN)
        result = filt.evaluate_all(board, [("d1d5", 0.9), ("e1e2", 0.1)])
        taking = next(e for e in result if e.uci == "d1d5")
        retreating = next(e for e in result if e.uci == "e1e2")
        assert taking.loss == 0
        assert retreating.loss > 0

    def test_filter_keeps_good_moves(self, filt):
        """При жёстком отсеве слабый ход отбрасывается."""
        board = chess.Board(TACTIC_FEN)
        candidates = [("d1d5", 0.9), ("e1e2", 0.1)]
        kept = filt.filter(board, candidates, strictness=1.0)
        assert any(e.uci == "d1d5" for e in kept)
        assert not any(e.uci == "e1e2" for e in kept)

    def test_no_filter_keeps_all(self, filt):
        """При strictness=0 фильтр ничего не убирает."""
        board = chess.Board(TACTIC_FEN)
        candidates = [("d1d5", 0.9), ("e1e2", 0.1)]
        kept = filt.filter(board, candidates, strictness=0.0)
        assert len(kept) == len(candidates)

    def test_never_returns_empty(self, filt):
        """Фильтр не может вернуть пустой список, если были кандидаты."""
        board = chess.Board(TACTIC_FEN)
        kept = filt.filter(board, [("e1e2", 0.5)], strictness=1.0)
        assert len(kept) == 1

    def test_context_manager(self):
        """Фильтр работает как контекстный менеджер."""
        with StockfishFilter(STOCKFISH_PATH, depth=4) as f:
            assert f is not None


@needs_model
@needs_stockfish
class TestHybridEngine:
    """Гибрид Nova + Stockfish."""

    @pytest.fixture(scope="class")
    def engine(self):
        eng = HumanLikeEngine(MODEL_PATH, STOCKFISH_PATH, filter_depth=6)
        yield eng
        eng.close()

    def test_filter_available(self, engine):
        assert engine.has_filter

    def test_analyzed_moves_sorted_by_loss(self, engine):
        """Отфильтрованный список отсортирован по возрастанию потери."""
        board = chess.Board(TACTIC_FEN)
        moves = engine.analyzed_moves(board, rating=1500, strictness=0.5)
        losses = [m.loss for m in moves]
        assert losses == sorted(losses)

    def test_choose_move_legal(self, engine):
        """Движок всегда возвращает легальный ход."""
        board = chess.Board()
        for _ in range(5):
            move = engine.choose_move(board, rating=1500, strictness=0.8)
            assert move in board.legal_moves

    def test_high_strictness_avoids_blunder(self, engine):
        """При strictness=1 в тактике выбирается взятие."""
        board = chess.Board(TACTIC_FEN)
        move = engine.choose_move(board, rating=800, strictness=1.0)
        assert move.uci() == "d1d5"

    def test_low_rating_keeps_blunders(self, engine):
        """На минимальном рейтинге зевок остаётся в списке ходов."""
        # Ng5 теряет ~727 сп — новичок его играет
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        low = engine.analyzed_moves(board, rating=800)
        high = engine.analyzed_moves(board, rating=2400)
        assert any(m.uci == "f3g5" for m in low), "на 800 зевок должен проходить"
        assert not any(m.uci == "f3g5" for m in high), "на 2400 зевок отсеивается"

    def test_strictness_none_uses_rating(self, engine):
        """Без явной строгости порог берётся из рейтинга."""
        board = chess.Board(TACTIC_FEN)
        low = engine.analyzed_moves(board, rating=800)
        high = engine.analyzed_moves(board, rating=2700)
        assert len(low) >= len(high)

    def test_engine_without_filter(self):
        """Без Stockfish движок работает в режиме чистой Nova."""
        eng = HumanLikeEngine(MODEL_PATH, None)
        try:
            assert not eng.has_filter
            move = eng.choose_move(chess.Board(), rating=1500)
            assert move in chess.Board().legal_moves
        finally:
            eng.close()

    def test_evaluate_position(self, engine):
        """Оценка позиции возвращает число."""
        cp = engine.evaluate_position(chess.Board(), depth=6)
        assert isinstance(cp, int)


@needs_model
@needs_stockfish
class TestStockfishMovesPerspective:
    """Регрессия: ходы Stockfish должны ранжироваться за того, кто ходит.

    Раньше оценка бралась с точки зрения белых при сортировке. При ходе
    чёрных это давало обратный порядок: движок предлагал ХУДШИЙ для них
    ход и отдавал фигуры бесплатно.

    Важно: используется ОДИН процесс Stockfish. Второй экземпляр в том же
    процессе python-chess ломает общий asyncio-loop, и движки падают с
    «engine event loop dead».
    """

    @pytest.fixture(scope="class")
    def engine(self):
        eng = HumanLikeEngine(MODEL_PATH, STOCKFISH_PATH, filter_depth=8)
        yield eng
        eng.close()

    POSITIONS = {
        "старт, белые ходят":
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "после 1.e4, чёрные ходят":
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "белый ферзь под боем ладьи":
            "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        "чёрный ферзь под боем ладьи":
            "3Rk3/8/8/3q4/8/8/8/4K3 b - - 0 1",
    }

    def _reference_best(self, engine, board, depth=16):
        """Лучший ход по Stockfish через тот же процесс."""
        import chess.engine

        with engine._filter._lock:
            result = engine._filter._engine.play(
                board, chess.engine.Limit(depth=depth))
        return result.move.uci()

    @pytest.mark.parametrize("label", list(POSITIONS))
    def test_top_move_matches_engine(self, engine, label):
        """Первый предложенный ход совпадает с лучшим ходом Stockfish."""
        fen = self.POSITIONS[label]
        board = chess.Board(fen)

        # Проверяем, что позиция вообще допустима для движка
        if board.status() != chess.STATUS_VALID:
            pytest.skip(f"позиция невозможна: {label}")

        mine = engine.stockfish_moves(board, top=3, depth=14)
        assert mine, f"нет ходов для: {label}"

        best = self._reference_best(engine, board)
        assert mine[0][0] == best, (
            f"{label}: предложен {mine[0][0]}, Stockfish считает лучшим {best}"
        )

    def test_probabilities_sum_to_one(self, engine):
        """Веса ходов нормированы."""
        pairs = engine.stockfish_moves(chess.Board(), top=5, depth=10)
        assert sum(p for _, p in pairs) == pytest.approx(1.0, abs=1e-6)

    def test_top_weight_dominates(self, engine):
        """Лучший ход получает заметно больший вес, чем остальные."""
        pairs = engine.stockfish_moves(chess.Board(), top=5, depth=12)
        assert pairs[0][1] > pairs[1][1]
