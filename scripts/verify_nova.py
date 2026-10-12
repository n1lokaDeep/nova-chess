import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import NovaEngine

n = NovaEngine(r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx")

print("=" * 68)
print("ПРОВЕРКА: действительно ли это НЕ Stockfish")
print("=" * 68)

# Позиция, где Stockfish и люди расходятся: 
# белые могут взять пешку, но лучше развивать фигуры
tests = [
    ("Старт", chess.STARTING_FEN),
    ("После 1.e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("После 1.d4 d5 2.c4", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 3"),
    ("Миттельшпиль", "r1bqkb1r/pp2pppp/2n2n2/3p4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
]

for name, fen in tests:
    b = chess.Board(fen)
    print()
    print(f"--- {name} ---")
    print(f"    {fen}")
    # разные рейтинги должны давать РАЗНЫЕ распределения
    for r in (800, 1600, 2400):
        top = n.predict(b, rating=r, top=3)
        s = "  ".join(f"{m.uci}={m.percent:.0f}%" for m in top)
        print(f"    {r}: {s}")
