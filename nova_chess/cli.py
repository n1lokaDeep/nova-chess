"""
CLI для Nova Chess.

Примеры:
    # позиция из FEN — что сыграл бы человек
    python -m nova_chess.cli moves --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --rating 1500

    # ход движка с фильтром качества
    python -m nova_chess.cli play --rating 1800 --strictness 0.7

    # сравнить чистую Nova и фильтрованную
    python -m nova_chess.cli compare

    # сыграть партию Nova против Nova
    python -m nova_chess.cli selfplay --moves 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

from .core import HumanLikeEngine

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "nova_v3b.onnx"

RATING_HELP = "Рейтинг игрока, 800–2700 (по умолчанию 1500)"
STYLE_HELP = "Ось стиля: 0 = гипермодернизм (Nf3, c4), 1 = классика (e4, d4)"
AGGR_HELP = "Агрессия: 0 = спокойно, 1 = тактика и атака"


def _build_engine(args: argparse.Namespace) -> HumanLikeEngine:
    model = Path(args.model)
    if not model.exists():
        sys.exit(f"Модель не найдена: {model}\nСкачайте nova_v3b.onnx с Hugging Face.")

    stockfish = getattr(args, "stockfish", None)
    if stockfish and not Path(stockfish).exists():
        print(f"Предупреждение: Stockfish не найден ({stockfish}), работаем без фильтра")
        stockfish = None

    return HumanLikeEngine(model, stockfish, filter_depth=getattr(args, "depth", 8))


def _describe(board: chess.Board) -> str:
    turn = "белые" if board.turn == chess.WHITE else "чёрные"
    extra = ""
    if board.is_check():
        extra = " (шах)"
    return f"Ход {turn}{extra}"


def cmd_moves(args: argparse.Namespace) -> int:
    """Показать ходы, которые сделал бы человек заданного рейтинга."""
    board = chess.Board(args.fen)
    engine = _build_engine(args)
    try:
        print(f"Позиция: {board.fen()}")
        print(f"{_describe(board)}")
        print(f"Рейтинг: {args.rating}   Стиль: {args.classical:.2f}   "
              f"Агрессия: {args.aggression:.2f}")
        print()

        if args.strictness > 0 and engine.has_filter:
            print(f"{'Ход':<8}{'Nova':>8}{'Stockfish':>12}   Оценка")
            print("-" * 46)
            for ev in engine.analyzed_moves(
                board, args.rating, args.classical, args.aggression,
                strictness=args.strictness, pool=args.top,
            ):
                tag = "лучший" if ev.is_best else f"потеря {ev.loss} сп"
                print(f"{ev.uci:<8}{ev.nova_probability*100:>7.1f}%"
                      f"{ev.centipawns:>+12}   {tag}")
        else:
            print(f"{'Ход':<8}{'Вероятность':>14}")
            print("-" * 24)
            for mv in engine.human_moves(
                board, args.rating, args.classical, args.aggression, top=args.top
            ):
                print(f"{mv.uci:<8}{mv.percent:>13.1f}%")
    finally:
        engine.close()
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Один ход движка с контролем качества."""
    board = chess.Board(args.fen)
    engine = _build_engine(args)
    try:
        move = engine.choose_move(
            board, args.rating, args.classical, args.aggression,
            strictness=args.strictness, sharpness=args.sharpness,
        )
        if move is None:
            print("Нет доступных ходов")
            return 1
        print(move.uci())
    finally:
        engine.close()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Сравнить чистую Nova и Nova со фильтром на тактической позиции."""
    board = chess.Board(args.fen)
    engine = _build_engine(args)
    if not engine.has_filter:
        sys.exit("Для сравнения нужен Stockfish (--stockfish ПУТЬ)")

    try:
        print(f"Позиция: {board.fen()}")
        print(_describe(board))
        print()

        print("Чистая Nova (как играет человек):")
        for mv in engine.human_moves(board, args.rating, top=5):
            print(f"  {mv.uci}  {mv.percent:5.1f}%")

        print()
        print(f"С фильтром качества (strictness={args.strictness:.1f}):")
        for ev in engine.analyzed_moves(board, args.rating, strictness=args.strictness):
            tag = "лучший" if ev.is_best else f"потеря {ev.loss} сп"
            print(f"  {ev.uci}  nova={ev.nova_probability*100:5.1f}%  "
                  f"оценка={ev.centipawns:+5d}  {tag}")
    finally:
        engine.close()
    return 0


def cmd_selfplay(args: argparse.Namespace) -> int:
    """Партия Nova против Nova с разными настройками."""
    engine = _build_engine(args)
    board = chess.Board()

    white = (args.rating, args.classical, args.aggression)
    black = (args.rating2 or args.rating, args.classical, args.aggression)

    try:
        for ply in range(args.moves):
            if board.is_game_over():
                break
            params = white if board.turn == chess.WHITE else black
            move = engine.choose_move(
                board, *params, strictness=args.strictness, sharpness=args.sharpness
            )
            if move is None:
                break
            san = board.san(move)
            board.push(move)
            number = board.fullmove_number
            prefix = f"{number}." if board.turn == chess.BLACK else f"{number}..."
            print(f"{prefix} {san}")

        print()
        if board.is_game_over():
            print(f"Результат: {board.result()} — {board.outcome().termination.name}")
        else:
            print(f"Остановлено на {ply if 'ply' in dir() else args.moves} полуходов")
    finally:
        engine.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nova-chess",
        description="Человеческий шахматный движок: Nova + фильтр Stockfish",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Путь к nova_v3b.onnx")
    parser.add_argument("--stockfish", default=None,
                        help="Путь к Stockfish (включает фильтр качества)")
    parser.add_argument("--depth", type=int, default=8,
                        help="Глубина анализа Stockfish (по умолчанию 8)")

    sub = parser.add_subparsers(dest="command", required=True)

    common = dict()
    for name, func, helptext in (
        ("moves", cmd_moves, "Ходы, которые сделал бы человек"),
        ("play", cmd_play, "Один ход движка"),
        ("compare", cmd_compare, "Сравнить чистую Nova и фильтрованную"),
        ("selfplay", cmd_selfplay, "Партия Nova против Nova"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--fen", default=chess.STARTING_FEN, help="Позиция в формате FEN")
        p.add_argument("--rating", type=int, default=1500, help=RATING_HELP)
        p.add_argument("--classical", type=float, default=0.5, help=STYLE_HELP)
        p.add_argument("--aggression", type=float, default=0.5, help=AGGR_HELP)
        p.set_defaults(func=func)

        if name in ("moves", "compare"):
            p.add_argument("--top", type=int, default=5, help="Сколько ходов показать")
            p.add_argument("--strictness", type=float, default=0.6,
                           help="Строгость фильтра, 0..1")
        if name == "play":
            p.add_argument("--strictness", type=float, default=0.6,
                           help="Строгость фильтра, 0..1")
            p.add_argument("--sharpness", type=float, default=3.0,
                           help="Насколько часто берётся самый вероятный ход")
        if name == "selfplay":
            p.add_argument("--moves", type=int, default=40, help="Число полуходов")
            p.add_argument("--rating2", type=int, default=None,
                           help="Рейтинг чёрных (если отличается)")
            p.add_argument("--strictness", type=float, default=0.6)
            p.add_argument("--sharpness", type=float, default=3.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
