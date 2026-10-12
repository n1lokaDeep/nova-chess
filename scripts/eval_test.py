import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import HumanLikeEngine

eng = HumanLikeEngine(
    r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
    r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
    filter_depth=8)

b = chess.Board()
print(f"{'#':<3}{'ход':<7}{'чей':<8}{'оценка':>8}{'дельта':>8}")
print("-" * 34)
prev = None
for i in range(12):
    cp = eng.evaluate_for_display(b, depth=12)
    d = "" if prev is None else f"{cp-prev:+d}"
    turn = "бел" if b.turn else "чёр"
    print(f"{i+1:<3}{'-':<7}{turn:<8}{cp:>8}{d:>8}")
    prev = cp
    mv = eng.choose_move(b, rating=1500)
    if mv is None: break
    b.push(mv)
print()
print("ВОПРОС: дельта постоянная? Если да — баг.")
eng.close()
