import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess, chess.engine
from nova_chess.core.filter import _score_to_cp, to_display_cp

SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 4, "Hash": 256})

# Позиция: конь под боем. Белый конь на f3, чёрная пешка g4 бьёт его? 
# Возьмём явное: белый конь на c3 под боем чёрного слона b4
tests = [
    ("Старт", chess.STARTING_FEN),
    ("Белый конь под боем", "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 5"),
    ("Простая: конь под боем пешки", "4k3/8/8/8/4p3/5N2/8/4K3 b - - 0 1"),
]

for name, fen in tests:
    b = chess.Board(fen)
    print("=" * 70)
    print(f"{name}")
    print(f"FEN: {fen}")
    print()
    info = eng.analyse(b, chess.engine.Limit(depth=16), multipv=3)
    if isinstance(info, dict): info = [info]
    for i in info:
        pv = i.get("pv", [])
        cp = _score_to_cp(i["score"], chess.WHITE)
        disp = to_display_cp(cp)
        san = b.san(pv[0]) if pv else "?"
        print(f"  {pv[0].uci() if pv else '?':<8} ({san:<8}) сырое cp={cp:>6}  на графике={disp:>6}")
    print()

print("=" * 70)
print("ПРОВЕРКА: как читается знак оценки")
print("=" * 70)
b = chess.Board("4k3/8/8/8/4p3/5N2/8/4K3 b - - 0 1")
print(f"Позиция: {b.fen()} (ход ЧЁРНЫХ, чёрная пешка e4 бьёт коня f3)")
print()
info = eng.analyse(b, chess.engine.Limit(depth=16))
sc = info["score"]
print(f"  сырой score.white()  = {sc.white().score() if not sc.white().is_mate() else 'M'}")
print(f"  _score_to_cp(WHITE)  = {_score_to_cp(sc, chess.WHITE)}")
print(f"  _score_to_cp(BLACK)  = {_score_to_cp(sc, chess.BLACK)}")
pv = info.get("pv", [])
if pv:
    print(f"  лучший ход: {pv[0].uci()} = {b.san(pv[0])}")
eng.quit()
