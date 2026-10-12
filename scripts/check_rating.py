import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import NovaEngine

n = NovaEngine(r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx")

# Твоя жалоба: между 800 и 2700 ходы не отличаются
print("=" * 70)
print("ПРОВЕРКА 1: отличаются ли ходы между 800 и 2700?")
print("=" * 70)

positions = [
    ("Старт", chess.STARTING_FEN),
    ("После 1.e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("1.d4 d5 2.c4", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 3"),
    ("Сицилианская", "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"),
]

for name, fen in positions:
    b = chess.Board(fen)
    print(f"\n--- {name} ---")
    p800 = n.move_probabilities(b, rating=800)
    p2700 = n.move_probabilities(b, rating=2700)
    t800 = max(p800.items(), key=lambda x: x[1])
    t2700 = max(p2700.items(), key=lambda x: x[1])
    print(f"  800:  {t800[0]} = {t800[1]*100:.1f}%")
    print(f"  2700: {t2700[0]} = {t2700[1]*100:.1f}%")
    if t800[0] == t2700[0]:
        print(f"  ОДИНАКОВЫЙ ТОП-ХОД!")
    else:
        print(f"  РАЗНЫЕ ХОДЫ")

print()
print("=" * 70)
print("ПРОВЕРКА 2: насколько сильно меняется распределение?")
print("=" * 70)
b = chess.Board()
p8 = n.move_probabilities(b, rating=800)
p27 = n.move_probabilities(b, rating=2700)
allm = sorted(set(list(p8.keys()) + list(p27.keys())))
print(f"{'ход':<8}{'800':>8}{'2700':>8}{'разница':>10}")
for m in sorted(allm, key=lambda x: -max(p8.get(x,0), p27.get(x,0)))[:8]:
    a, c = p8.get(m,0)*100, p27.get(m,0)*100
    print(f"{m:<8}{a:>7.1f}%{c:>7.1f}%{c-a:>+9.1f}%")
