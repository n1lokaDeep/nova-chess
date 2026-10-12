"""Тесты кодирования позиций."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from nova_chess.core.encoding import (
    MAX_RATING,
    MIN_RATING,
    fen_to_planes,
    make_conditioning,
    normalize_rating,
)


class TestFenToPlanes:
    """Кодирование FEN в тензор плоскостей."""

    def test_shape(self):
        """Тензор всегда (18, 8, 8)."""
        planes = fen_to_planes(chess.STARTING_FEN)
        assert planes.shape == (18, 8, 8)
        assert planes.dtype == np.float32

    def test_starting_position_piece_count(self):
        """На старте 32 фигуры — сумма по плоскостям фигур равна 32."""
        planes = fen_to_planes(chess.STARTING_FEN)
        assert planes[:12].sum() == 32

    def test_white_to_move_plane(self):
        """Плоскость 12 = 1 когда ход белых, 0 когда чёрных."""
        white = fen_to_planes(chess.STARTING_FEN)
        black = fen_to_planes(chess.STARTING_FEN.replace(" w ", " b "))
        assert white[12].sum() == 64
        assert black[12].sum() == 0

    def test_castling_planes(self):
        """Права рокировки отражаются в плоскостях 13–16."""
        planes = fen_to_planes(chess.STARTING_FEN)
        assert planes[13].sum() == 64  # K
        assert planes[14].sum() == 64  # Q
        assert planes[15].sum() == 64  # k
        assert planes[16].sum() == 64  # q

        none = fen_to_planes("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        assert none[13].sum() == 0
        assert none[16].sum() == 0

    def test_rank_orientation(self):
        """Белые пешки стоят на 2-м ряду, что соответствует row=1."""
        planes = fen_to_planes(chess.STARTING_FEN)
        # Плоскость 0 — белые пешки
        assert planes[0, 1, :].sum() == 8
        # Плоскость 6 — чёрные пешки, row=6
        assert planes[6, 6, :].sum() == 8

    def test_en_passant_plane(self):
        """Поле взятия на проходе кодируется в плоскости 17."""
        fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
        planes = fen_to_planes(fen)
        assert planes[17].sum() == 1
        # c6 → file index 2
        assert planes[17, 0, 2] == 1.0

    def test_kings_only(self):
        """Минимальная позиция кодируется без ошибок."""
        planes = fen_to_planes("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
        assert planes[:12].sum() == 2

    def test_invalid_fen_raises(self):
        """Некорректный FEN вызывает ValueError."""
        with pytest.raises(ValueError):
            fen_to_planes("слишком короткий")
        with pytest.raises(ValueError):
            fen_to_planes("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")


class TestRatingNormalization:
    """Нормализация рейтинга."""

    @pytest.mark.parametrize("rating,expected", [
        (800, 0.0),
        (2700, 1.0),
        (1750, 0.5),
    ])
    def test_bounds(self, rating, expected):
        """Границы дают 0 и 1, середина — 0.5."""
        assert normalize_rating(rating) == pytest.approx(expected, abs=1e-6)

    def test_clamping(self):
        """Значения вне диапазона обрезаются."""
        assert normalize_rating(100) == 0.0
        assert normalize_rating(9999) == 1.0

    def test_monotonic(self):
        """Рост рейтинга даёт рост нормированного значения."""
        values = [normalize_rating(r) for r in range(MIN_RATING, MAX_RATING, 100)]
        assert values == sorted(values)


class TestConditioning:
    """Вектор условий для модели."""

    def test_shape_and_dtype(self):
        cond = make_conditioning(1500, 0.5, 0.5)
        assert cond.shape == (1, 3)
        assert cond.dtype == np.float32

    def test_values(self):
        cond = make_conditioning(1750, 0.0, 1.0)
        assert cond[0, 0] == pytest.approx(0.5, abs=1e-6)
        assert cond[0, 1] == pytest.approx(0.0)
        assert cond[0, 2] == pytest.approx(1.0)
