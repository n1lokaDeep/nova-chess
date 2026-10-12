"""Тесты проверки позиций на стороне сервера.

Именно здесь ловится позиция, на которой Stockfish падает:
python-chess формально разбирает FEN, но позиция невозможна
(лишние фигуры, пешка на последней горизонтали и т. п.).
"""

from __future__ import annotations

import chess
import pytest

from nova_chess.web.server import _describe_status, _safe_board


class TestSafeBoard:
    """Проверка позиции перед отправкой в движок."""

    def test_valid_start(self):
        board = _safe_board(chess.STARTING_FEN)
        assert board.turn == chess.WHITE

    def test_valid_midgame(self):
        fen = "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 3"
        board = _safe_board(fen)
        assert board.turn == chess.BLACK

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            _safe_board("это не FEN")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _safe_board("")

    def test_rejects_too_few_rows(self):
        """В поле доски должно быть ровно 8 горизонталей."""
        with pytest.raises(ValueError):
            _safe_board("rnbqkbnr/pppppppp/8/8 w KQkq - 0 1")

    def test_accepts_bare_board_fen(self):
        """Голое поле доски python-chess принимает, подставляя остальное.

        Это осознанное поведение библиотеки, а не ошибка: фиксируем его
        тестом, чтобы не удивляться в будущем.
        """
        board = _safe_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        assert board.turn == chess.WHITE

    def test_rejects_too_many_pieces(self):
        """Позиция с лишними фигурами — та самая, что убивала Stockfish."""
        bad = "r1bqkb1r/ppp2ppp/2n2n2/3pp1b1/3P4/2N1B2N/PPP2PPP/R2QKB1R w KQkq - 0 6"
        raw = chess.Board(bad)
        assert raw.status() != chess.STATUS_VALID   # python-chess разбирает

        with pytest.raises(ValueError, match="невозможна"):
            _safe_board(bad)                        # а наш код отклоняет

    def test_rejects_no_kings(self):
        with pytest.raises(ValueError, match="невозможна"):
            _safe_board("8/8/8/8/8/8/8/8 w - - 0 1")

    def test_rejects_pawn_on_backrank(self):
        # Пешка на 8-й горизонтали невозможна: строка доски идёт от 8-й вниз
        with pytest.raises(ValueError, match="невозможна"):
            _safe_board("P3k3/8/8/8/8/8/8/4K3 w - - 0 1")

    def test_allows_pawn_on_seventh(self):
        """Пешка на 7-й горизонтали — обычная позиция перед превращением."""
        board = _safe_board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        assert board.status() == chess.STATUS_VALID

    def test_error_message_is_meaningful(self):
        """Сообщение должно объяснять причину, а не просто «ошибка»."""
        with pytest.raises(ValueError) as excinfo:
            _safe_board("8/8/8/8/8/8/8/8 w - - 0 1")
        assert len(str(excinfo.value)) > 20


class TestDescribeStatus:
    """Человекочитаемое объяснение кода статуса."""

    def test_valid_status(self):
        text = _describe_status(chess.STATUS_VALID)
        assert text    # для валидного тоже что-то возвращаем

    def test_no_white_king(self):
        text = _describe_status(chess.STATUS_NO_WHITE_KING)
        assert "белого короля" in text

    def test_no_black_king(self):
        text = _describe_status(chess.STATUS_NO_BLACK_KING)
        assert "чёрного короля" in text

    def test_pawns_on_backrank(self):
        text = _describe_status(chess.STATUS_PAWNS_ON_BACKRANK)
        assert "последней горизонтали" in text

    def test_too_many_pieces(self):
        text = _describe_status(chess.STATUS_TOO_MANY_WHITE_PIECES)
        assert "белых фигур" in text

    def test_combines_multiple_reasons(self):
        combined = chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING
        text = _describe_status(combined)
        assert "белого короля" in text
        assert "чёрного короля" in text
        assert "," in text

    def test_unknown_code_falls_back(self):
        """Неизвестный код не должен ронять функцию."""
        text = _describe_status(1 << 20)
        assert text
