import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import HumanLikeEngine

eng = HumanLikeEngine(
    r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
    r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
    filter_depth=8)

# Позиция где ХОД ЧЁРНЫХ: чёрный конь под боем, надо его спасти
# Белая пешка d5 атакует коня c6. Чёрные должны что-то сделать
pos = chess.Board("r1bqkb1r/ppp1pppp/2n5/3p4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 0 4")
print("FEN:", pos.fen())
print("Ход:", "чёрные" if not pos.turn else "белые")
print()

print("=== Stockfish напрямую (эталон) ===")
import chess.engine
e = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe")
best = e.play(pos, chess.engine.Limit(depth=14))
print("  лучший ход:", best.move.uci())
info = e.analyse(pos, chess.engine.Limit(depth=14), multipv=5)
if isinstance(info, dict): info = [info]
for i in info:
    pv = i.get("pv", [])
    sc = i["score"].white()
    print(f"    {pv[0].uci() if pv else '?':<8} score.white()={sc.score() if not sc.is_mate() else 'M'}")
e.quit()

print()
print("=== МОЙ stockfish_moves ===")
pairs = eng.stockfish_moves(pos, top=5)
for uci, p in pairs:
    print(f"    {uci:<8} вес={p:.3f}")

print()
print("=== СРАВНЕНИЕ ===")
if pairs:
    my_best = pairs[0][0]
    print(f"  Stockfish напрямую: {best.move.uci()}")
    print(f"  Мой лучший:         {my_best}")
    print("  СОВПАДАЕТ" if my_best == best.move.uci() else "  ❌ РАСХОЖДЕНИЕ — ЭТО БАГ")
eng.close()
