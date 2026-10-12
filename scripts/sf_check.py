import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
import chess.engine

SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 4, "Hash": 256})

# Позиция: чёрный ферзь под боем у белой ладьи
pos = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
print("Позиция:", pos.fen())
print("Белая ладья d1, чёрный ферзь d5 — есть взятие d1xd5")
print()

print("=== Multipv анализ Stockfish (глубина 14) ===")
info = eng.analyse(pos, chess.engine.Limit(depth=14), multipv=5)
if isinstance(info, dict): info = [info]
for i, line in enumerate(info):
    pv = line.get("pv", [])
    sc = line["score"].white()
    cp = sc.score() if not sc.is_mate() else f"M{sc.mate()}"
    first = pv[0].uci() if pv else "?"
    print(f"  {i+1}. {first}   оценка {cp}")

print()
print("=== Простой поиск лучшего хода ===")
best = eng.play(pos, chess.engine.Limit(depth=14))
print("  play() ->", best.move.uci())

print()
print("=== Проверка: реально ли d1d5 легален и выгоден? ===")
m = chess.Move.from_uci("d1d5")
print("  легален:", m in pos.legal_moves)
after = pos.copy(); after.push(m)
print("  после d1d5 материал: белые R, чёрные ничего")
info2 = eng.analyse(after, chess.engine.Limit(depth=12))
sc = info2["score"].white()
print("  оценка после взятия:", sc.score() if not sc.is_mate() else f"M{sc.mate()}")
eng.quit()
