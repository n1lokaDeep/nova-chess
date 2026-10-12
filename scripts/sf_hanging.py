import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess, chess.engine
from nova_chess.core.filter import _score_to_cp, to_display_cp

SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 4, "Hash": 256})

print("=" * 72)
print("КЛЮЧЕВОЙ ВОПРОС: как оценивается позиция, где ФИГУРА ПОД БОЕМ")
print("=" * 72)

# Классика: белый конь c3 атакован чёрной пешкой b4, защиты нет
pos = chess.Board("r1bqkbnr/p1pppppp/1pn5/8/1p6/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 4")
print(f"\nFEN: {pos.fen()}")
print("Белый конь c3 атакован пешкой b4, защиты нет (-300 сп)")
print()

print("Что говорит Stockfish (глубина 14):")
info = eng.analyse(pos, chess.engine.Limit(depth=14), multipv=3)
if isinstance(info, dict): info = [info]
for i in info:
    pv = i.get("pv", [])
    cp = _score_to_cp(i["score"], chess.WHITE)
    san = pos.san(pv[0]) if pv else "?"
    print(f"  {pv[0].uci():<8} {san:<8} cp={cp:>6}  график={to_display_cp(cp):>6}")
eng.quit()
