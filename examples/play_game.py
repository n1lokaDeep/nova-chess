#!/usr/bin/env python
"""
Партия против движка прямо в терминале.

Пример:
    python examples/play_game.py --rating 1600
    python examples/play_game.py --rating 1200 --color black

Ходы вводятся в формате UCI (e2e4) или SAN (e4, Nf3).
Требуется модель Nova в models/nova_v3b.onnx (см. README).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nova_chess.core import HumanLikeEngine  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "nova_v3b.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rating", type=int, default=1500,
                        help="рейтинг движка, 800-2700")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="за кого играет человек")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--stockfish", default=os.environ.get("NOVA_STOCKFISH_PATH"))
    return parser.parse_args()


def read_move(board: chess.Board, text: str) -> chess.Move | None:
    """Разбирает ход пользователя: сначала UCI, потом SAN."""
    text = text.strip()
    if not text:
        return None

    # UCI — самый однозначный вариант
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # SAN — как в записи партии (Nf3, e4, O-O)
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def show(board: chess.Board) -> None:
    print()
    print(board.unicode(borders=True, empty_square="·"))
    print()


def main() -> int:
    args = parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"Модель не найдена: {model}")
        print("Скачайте nova_v3b.onnx и nova_v3b.onnx.data (см. README).")
        return 1

    stockfish = args.stockfish if args.stockfish and Path(args.stockfish).exists() else None
    human_color = chess.WHITE if args.color == "white" else chess.BLACK

    board = chess.Board()

    with HumanLikeEngine(model, stockfish) as engine:
        print(f"Вы играете {'белыми' if human_color else 'чёрными'} "
              f"против движка рейтинга {args.rating}.")
        print("Ход: UCI (e2e4) или SAN (e4, Nf3). Выход — exit.")

        while not board.is_game_over():
            show(board)

            if board.turn == human_color:
                text = input("ваш ход> ")
                if text.strip().lower() in ("exit", "quit", "q"):
                    print("Партия прервана.")
                    return 0

                move = read_move(board, text)
                if move is None:
                    print("Не понял ход. Примеры: e2e4, Nf3, O-O")
                    continue
            else:
                print("движок думает…")
                move = engine.choose_move(board, rating=args.rating)
                if move is None:
                    break
                print(f"движок: {board.san(move)}")

            board.push(move)

        show(board)
        if board.is_game_over():
            reason = board.outcome().termination.name
            print(f"Результат: {board.result()}  ({reason})")
        else:
            print("Партия остановлена.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
