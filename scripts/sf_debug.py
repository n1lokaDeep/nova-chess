import chess, chess.engine, traceback
SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"

print("=== ТЕСТ 1: простой запуск ===")
try:
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    print("  запущен OK")
    print("  id:", eng.id.get("name"))
    r = eng.play(chess.Board(), chess.engine.Limit(depth=10))
    print("  ход:", r.move.uci())
    eng.quit()
    print("  закрыт OK")
except Exception as e:
    print("  ОШИБКА:", type(e).__name__, e)
    traceback.print_exc()

print()
print("=== ТЕСТ 2: configure (как в моём коде) ===")
try:
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 4, "Hash": 256})
    print("  configure OK")
    r = eng.play(chess.Board(), chess.engine.Limit(depth=10))
    print("  ход:", r.move.uci())
    eng.quit()
except Exception as e:
    print("  ОШИБКА:", type(e).__name__, e)

print()
print("=== ТЕСТ 3: multipv (как в stockfish_moves) ===")
try:
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 4, "Hash": 256})
    info = eng.analyse(chess.Board(), chess.engine.Limit(depth=10), multipv=5)
    print("  multipv OK, линий:", len(info) if isinstance(info, list) else 1)
    eng.quit()
except Exception as e:
    print("  ОШИБКА:", type(e).__name__, e)
    traceback.print_exc()
