import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess, math
from nova_chess.core import HumanLikeEngine

eng = HumanLikeEngine(
    r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
    r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
    filter_depth=8)

print("=" * 60)
print("ПРОВЕРКА: стабильность оценки на разных глубинах")
print("=" * 60)
b = chess.Board()
print(f"\n{'глубина':>8}{'оценка':>10}")
for d in (8, 10, 12, 14, 16, 18, 20):
    cp = eng.evaluate_for_display(b, depth=d, multipv=1)
    print(f"{d:>8}{cp:>10}")

print()
print("=" * 60)
print("ПРОВЕРКА: скачет ли знак на серии ходов (глубина 18)")
print("=" * 60)
b = chess.Board()
print(f"\n{'#':<3}{'чей':<7}{'оценка':>8}")
print("-" * 20)
signs = []
for i in range(10):
    cp = eng.evaluate_for_display(b, depth=18)
    signs.append(cp >= 0)
    print(f"{i+1:<3}{('бел' if b.turn else 'чёр'):<7}{cp:>8}")
    mv = eng.choose_move(b, rating=1500)
    if mv is None: break
    b.push(mv)

flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
print(f"\nСмен знака: {flips} из {len(signs)-1}")
if flips > len(signs) // 2:
    print("ПЛОХО: знак скачет постоянно")
else:
    print("OK: знак меняется только когда позиция реально меняется")
eng.close()
