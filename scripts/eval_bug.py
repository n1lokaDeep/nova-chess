import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import HumanLikeEngine

eng = HumanLikeEngine(
    r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
    r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
    filter_depth=8)

pos = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
print("Позиция:", pos.fen())
print()
print("=== evaluate_position (сырая, за белых) ===")
print("  ", eng.evaluate_position(pos, depth=14))
print()
print("=== evaluate_for_display (для графика) ===")
for mpv in (1, 3, 5):
    print(f"  multipv={mpv}: ", eng.evaluate_for_display(pos, depth=14, multipv=mpv))
print()
print("=== Что реально возвращает multipv=3 из Stockfish ===")
import chess.engine
from nova_chess.core.filter import _score_to_cp
with eng._filter._lock:
    raw = eng._filter._engine.analyse(pos, chess.engine.Limit(depth=14), multipv=3)
infos = raw if isinstance(raw, list) else [raw]
for i in infos:
    pv = i.get("pv", [])
    cp = _score_to_cp(i["score"], chess.WHITE)
    print(f"  линия: {pv[0].uci() if pv else '?'}  cp={cp}")
eng.close()
