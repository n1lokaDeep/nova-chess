"""Тесты движка Nova.

Тесты, требующие модель, помечены `model` и пропускаются,
если файл `nova_v3b.onnx` не найден.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from nova_chess.core.engine import (
    BOARD_SQUARES,
    NovaEngine,
    _move_to_index,
    index_to_uci,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "nova_v3b.onnx"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"модель не найдена: {MODEL_PATH}",
)


class TestMoveIndexing:
    """Преобразование ходов в индексы пространства Nova."""

    def test_normal_move(self):
        """Обычный ход: from * 64 + to."""
        move = chess.Move.from_uci("e2e4")
        expected = chess.E2 * BOARD_SQUARES + chess.E4
        assert _move_to_index(move) == expected

    @pytest.mark.parametrize("uci,offset", [
        ("e7e8n", 4096),
        ("e7e8b", 8192),
        ("e7e8r", 12288),
        ("e7e8q", 0),
    ])
    def test_promotions(self, uci, offset):
        """Превращения смещены в отдельные блоки."""
        move = chess.Move.from_uci(uci)
        base = move.from_square * BOARD_SQUARES + move.to_square
        assert _move_to_index(move) == base + offset

    def test_index_roundtrip(self):
        """index_to_uci восстанавливает ход (с позицией для превращений)."""
        # Позиция, где возможно превращение белой пешки e7
        board = chess.Board("8/4P3/8/8/8/8/8/K6k w - - 0 1")
        for uci in ("e7e8q", "e7e8n", "e7e8r", "e7e8b"):
            move = chess.Move.from_uci(uci)
            assert index_to_uci(_move_to_index(move), board) == uci

    def test_index_roundtrip_normal(self):
        """Обычные ходы восстанавливаются без позиции."""
        for uci in ("e2e4", "g1f3", "a1h8"):
            move = chess.Move.from_uci(uci)
            assert index_to_uci(_move_to_index(move)) == uci

    def test_queen_promotion_needs_board(self):
        """Без позиции превращение в ферзя неотличимо от обычного хода."""
        move = chess.Move.from_uci("e7e8q")
        # Без board суффикс не восстанавливается — это задокументированное поведение
        assert index_to_uci(_move_to_index(move)) == "e7e8"

    def test_index_in_range(self):
        """Индексы лежат в пространстве 16384."""
        board = chess.Board()
        for move in board.legal_moves:
            idx = _move_to_index(move)
            assert 0 <= idx < 16384


@needs_model
class TestNovaEngine:
    """Тесты с реальной моделью."""

    @pytest.fixture(scope="class")
    def engine(self):
        eng = NovaEngine(MODEL_PATH, intra_op_threads=2)
        yield eng

    def test_probabilities_sum_to_one(self, engine):
        """Вероятности легальных ходов нормированы."""
        probs = engine.move_probabilities(chess.Board(), rating=1500)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)

    def test_only_legal_moves(self, engine):
        """В распределении только легальные ходы."""
        board = chess.Board()
        legal = {m.uci() for m in board.legal_moves}
        probs = engine.move_probabilities(board, rating=1500)
        assert set(probs) == legal

    def test_starting_position_reasonable(self, engine):
        """На старте e4 или d4 — самые вероятные ходы людей."""
        top = engine.predict(chess.Board(), rating=1500, top=2)
        assert top[0].uci in ("e2e4", "d2d4")

    def test_rating_changes_prediction(self, engine):
        """Разные рейтинги дают разные распределения."""
        board = chess.Board()
        low = engine.move_probabilities(board, rating=800)
        high = engine.move_probabilities(board, rating=2500)
        assert low != high
        # c4 чаще играют высокорейтинговые
        assert high.get("c2c4", 0) > low.get("c2c4", 0)

    def test_top_sorted_descending(self, engine):
        """predict возвращает список по убыванию вероятности."""
        moves = engine.predict(chess.Board(), rating=1500, top=5)
        probs = [m.probability for m in moves]
        assert probs == sorted(probs, reverse=True)
        assert len(moves) == 5

    def test_sample_move_is_legal(self, engine):
        """Семплированный ход всегда легален."""
        board = chess.Board()
        for _ in range(20):
            move = engine.sample_move(board, rating=1500)
            assert move in board.legal_moves

    def test_sample_move_no_garbage(self, engine):
        """Хвост распределения отсекается: b2b3 почти не выпадает."""
        board = chess.Board()
        picks = [engine.sample_move(board, rating=1500).uci() for _ in range(60)]
        # b2b3 имеет вероятность ~0.1%, при отсеве не должен появляться
        assert picks.count("b2b3") <= 1

    def test_checkmate_position(self, engine):
        """В позиции мата в 1 высокий рейтинг чаще находит его."""
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        probs_high = engine.move_probabilities(board, rating=2700)
        probs_low = engine.move_probabilities(board, rating=800)
        assert probs_high.get("a1a8", 0) > probs_low.get("a1a8", 0)
