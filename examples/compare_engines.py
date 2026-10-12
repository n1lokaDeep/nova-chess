#!/usr/bin/env python
"""
Сравнение движков на одной позиции.

Показывает разницу между Nova и Maia-2: как меняется выбор хода
с ростом рейтинга у каждой модели.

Пример:
    python examples/compare_engines.py
    python examples/compare_engines.py --fen "<FEN>"
    python examples/compare_engines.py --top 5

Maia-2 необязательна: если пакет не установлен, сравнение идёт
только по Nova, и об этом сообщается.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nova_chess.core import NovaEngine  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "nova_v3b.onnx"

# Точки, в которых сравниваем поведение
NOVA_RATINGS = [800, 1400, 2000, 2700]
MAIA_RATINGS = [1100, 1400, 1700, 1900]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Ферзевый гамбит: на этой позиции разница между уровнями заметна
    # у обеих моделей — новички берут пешку, опытные играют теорию
    parser.add_argument("--fen", default="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/"
                                        "PP2PPPP/RNBQKBNR b KQkq - 0 3")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    return parser.parse_args()


def describe(board: chess.Board, moves: list[tuple[str, float]]) -> str:
    """Строка вида «e2e4 56% · d2d4 24%» с расшифровкой SAN."""
    parts = []
    for uci, prob in moves:
        try:
            san = board.san(chess.Move.from_uci(uci))
        except ValueError:
            san = uci
        parts.append(f"{san} {prob * 100:.0f}%")
    return "   ".join(parts)


def main() -> int:
    args = parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"Модель не найдена: {model}")
        print("Скачайте nova_v3b.onnx и nova_v3b.onnx.data (см. README).")
        return 1

    board = chess.Board(args.fen)

    # Невозможную позицию движки обработают молча, но результат
    # будет бессмысленным — лучше предупредить сразу
    if board.status() != chess.STATUS_VALID:
        print(f"Внимание: позиция невозможна (код {board.status()}).")
        print("Движки её обработают, но результат может быть бессмысленным.")
        print()

    print(f"Позиция: {board.fen()}")
    print(f"Ход:     {'белые' if board.turn else 'чёрные'}")
    print()

    # ---- Nova ----
    print("=" * 66)
    print("NOVA — обучена на 500 млн партий Lichess (800–2700)")
    print("=" * 66)

    nova = NovaEngine(model)
    nova_rows = []
    for rating in NOVA_RATINGS:
        top = nova.predict(board, rating=rating, top=args.top)
        moves = [(m.uci, m.probability) for m in top]
        nova_rows.append((rating, moves))
        print(f"  {rating:>4}: {describe(board, moves)}")

    # ---- Maia-2 ----
    print()
    print("=" * 66)
    print("MAIA-2 — обучена по рейтинговым диапазонам (1100–1900)")
    print("=" * 66)

    try:
        from nova_chess.core import MaiaEngine

        print("  загружаю модель…")
        maia = MaiaEngine()
        for rating in MAIA_RATINGS:
            top = maia.predict(board, elo_self=rating, elo_oppo=rating, top=args.top)
            moves = [(m.uci, m.probability) for m in top]
            print(f"  {rating:>4}: {describe(board, moves)}")
    except ImportError:
        print("  Maia-2 не установлена — сравнение пропущено.")
        print("  Установка: pip install maia2")
    except Exception as exc:
        print(f"  Maia-2 недоступна: {exc}")

    # ---- Вывод ----
    print()
    print("=" * 66)
    print("НА ЧТО СМОТРЕТЬ")
    print("=" * 66)
    print("  • Как меняется доля с ростом рейтинга внутри каждой модели.")
    print("    У Nova разница обычно меньше: в открытом релизе нет")
    print("    калибровочного слоя (подробности в README).")
    print("  • Совпадают ли модели по первому ходу. Часто да, но не всегда.")
    print("  • Доля — это частота выбора людьми, а не сила хода.")
    print("    Ход с 60% может быть объективно плохим — его просто")
    print("    часто играют на этом уровне.")
    print()
    print("  Заметьте: конкретные числа зависят от позиции. На одних")
    print("  Nova меняется сильнее, на других — Maia-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
