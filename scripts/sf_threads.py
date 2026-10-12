import chess, chess.engine, traceback
SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
FEN = "r1bqkb1r/ppp2ppp/2n2n2/3pp1b1/3P4/2N1B2N/PPP2PPP/R2QKB1R w KQkq - 0 6"
b = chess.Board(FEN)

print("=== ТЕСТ А: без Threads, multipv=5 ===")
try:
    e = chess.engine.SimpleEngine.popen_uci(SF)
    info = e.analyse(b, chess.engine.Limit(depth=14), multipv=5)
    print("  OK, линий:", len(info) if isinstance(info,list) else 1)
    e.quit()
except Exception as ex:
    print("  УПАЛ:", type(ex).__name__, ex)

print()
print("=== ТЕСТ Б: Threads=4, multipv=5 ===")
try:
    e = chess.engine.SimpleEngine.popen_uci(SF)
    e.configure({"Threads": 4, "Hash": 256})
    info = e.analyse(b, chess.engine.Limit(depth=14), multipv=5)
    print("  OK, линий:", len(info) if isinstance(info,list) else 1)
    e.quit()
except Exception as ex:
    print("  УПАЛ:", type(ex).__name__, ex)

print()
print("=== ТЕСТ В: Threads=4 на СТАРТОВОЙ позиции ===")
try:
    e = chess.engine.SimpleEngine.popen_uci(SF)
    e.configure({"Threads": 4, "Hash": 256})
    info = e.analyse(chess.Board(), chess.engine.Limit(depth=14), multipv=5)
    print("  OK, линий:", len(info) if isinstance(info,list) else 1)
    e.quit()
except Exception as ex:
    print("  УПАЛ:", type(ex).__name__, ex)

print()
print("=== ТЕСТ Г: Threads=4, БЕЗ multipv, на этом FEN ===")
try:
    e = chess.engine.SimpleEngine.popen_uci(SF)
    e.configure({"Threads": 4, "Hash": 256})
    info = e.analyse(b, chess.engine.Limit(depth=14))
    print("  OK, оценка:", info["score"].white())
    e.quit()
except Exception as ex:
    print("  УПАЛ:", type(ex).__name__, ex)
