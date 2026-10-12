import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import HumanLikeEngine
from nova_chess.core.filter import rating_to_strictness, loss_budget

print("=" * 72)
print("КАЛИБРОВКА: рейтинг -> строгость -> допустимая потеря")
print("=" * 72)
for r in (800, 1200, 1600, 2000, 2400, 2700):
    s = rating_to_strictness(r)
    b = loss_budget(s)
    print(f"  {r:>4} эло -> строгость {s:.2f} -> потеря до {b:>3} сп")

print()
print("=" * 72)
print("ПРОВЕРКА: отличается ли игра между рейтингами?")
print("=" * 72)

eng = HumanLikeEngine(
    r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
    r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
    filter_depth=8,
)

# Позиция с заманчивым, но плохим ходом Ng5
pos = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
print(f"\nПозиция: {pos.fen()}")
print("Тут Ng5 теряет 727 сп — настоящий зевок\n")

print(f"{'эло':>5} {'строгость':>10} {'ходов':>7}   какие ходы прошли")
for r in (800, 1200, 1600, 2000, 2400, 2700):
    ev = eng.analyzed_moves(pos, rating=r)
    moves = "  ".join(e.uci for e in ev[:6])
    st = rating_to_strictness(r)
    ng5 = "Ng5 ЕСТЬ" if any(e.uci == "f3g5" for e in ev) else "Ng5 отсеян"
    print(f"{r:>5} {st:>10.2f} {len(ev):>7}   {moves}")
    print(f"{'':>5} {'':>10} {'':>7}   -> {ng5}")

eng.close()
