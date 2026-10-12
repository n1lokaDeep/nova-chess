#!/usr/bin/env python
"""
Разбор позиции: что сыграли бы люди заданного рейтинга.

Пример:
    python examples/analyze_position.py
    python examples/analyze_position.py --rating 2400
    python examples/analyze_position.py --fen "8/8/8/3q4/8/8/8/3RK2k w - - 0 1"

Требуется модель Nova в models/nova_v3b.onnx (см. README).
Stockfish не обязателен, но с ним появляется оценка ходов.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess

# Пакет лежит в корне репозитория — добавляем его в пути импорта,
# чтобы пример запускался прямо из папки examples
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nova_chess.core import HumanLikeEngine  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "nova_v3b.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen", default=chess.STARTING_FEN,
                        help="позиция в формате FEN")
    parser.add_argument("--rating", type=int, default=1500,
                        help="рейтинг игрока, 800-2700")
    parser.add_argument("--top", type=int, default=5,
                        help="сколько ходов показать")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="путь к nova_v3b.onnx")
    parser.add_argument("--stockfish", default=os.environ.get("NOVA_STOCKFISH_PATH"),
                        help="путь к Stockfish (иначе без оценки)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"Модель не найдена: {model}")
        print("Скачайте nova_v3b.onnx и nova_v3b.onnx.data (см. README).")
        return 1

    # Stockfish подключаем только если файл реально существует:
    # иначе движок работает без оценки качества ходов
    stockfish = args.stockfish if args.stockfish and Path(args.stockfish).exists() else None

    board = chess.Board(args.fen)
    print(f"Позиция:  {board.fen()}")
    print(f"Ход:      {'белые' if board.turn else 'чёрные'}")
    print(f"Рейтинг:  {args.rating}")
    print(f"Stockfish: {'подключён' if stockfish else 'нет (без оценки)'}")
    print()

    with HumanLikeEngine(model, stockfish) as engine:
        if stockfish:
            print(f"{'Ход':<8}{'люди':>8}{'оценка':>10}   Комментарий")
            print("-" * 52)
            for ev in engine.analyzed_moves(board, rating=args.rating,
                                            strictness=0.0, pool=args.top):
                san = board.san(chess.Move.from_uci(ev.uci))
                note = "лучший" if ev.is_best else f"потеря {ev.loss} сп"
                print(f"{ev.uci:<8}{ev.nova_probability*100:>7.1f}%"
                      f"{ev.centipawns:>+10}   {san} — {note}")
        else:
            print(f"{'Ход':<8}{'люди':>8}   Сан")
            print("-" * 34)
            for mv in engine.human_moves(board, rating=args.rating, top=args.top):
                san = board.san(chess.Move.from_uci(mv.uci))
                print(f"{mv.uci:<8}{mv.percent:>7.1f}%   {san}")

    print()
    print("Значения — доля людей этого рейтинга, выбравших такой ход.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
