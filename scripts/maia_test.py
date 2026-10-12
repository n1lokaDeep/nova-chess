import sys

sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess

from nova_chess.core.maia import MaiaEngine

print("Загружаю Maia-2...")
m = MaiaEngine()
print("OK")
print()
print("=" * 66)
print("MAIA-2: разница между рейтингами")
print("=" * 66)

pos = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
print(f"\nПозиция: {pos.fen()}\n")
for r in (1100, 1300, 1500, 1700, 1900):
    top = m.predict(pos, elo_self=r, elo_oppo=r, top=4)
    s = "  ".join(f"{x.uci}={x.percent:.0f}%" for x in top)
    print(f"  {r}: {s}")

print()
print("Стартовая позиция:")
b = chess.Board()
for r in (1100, 1500, 1900):
    top = m.predict(b, elo_self=r, elo_oppo=r, top=4)
    s = "  ".join(f"{x.uci}={x.percent:.0f}%" for x in top)
    print(f"  {r}: {s}")

print()
print("Ожидаемый счёт (насколько сильнее себя против равного):")
for r in (1100, 1500, 1900):
    sc = m.expected_score(chess.Board(), elo_self=r, elo_oppo=1500)
    print(f"  {r} против 1500: {sc:.1%}")
