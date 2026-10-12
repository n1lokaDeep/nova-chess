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
