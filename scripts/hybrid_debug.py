import sys, os, traceback
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
print("Рабочая директория:", os.getcwd())
print()
print("=== ЗАПУСК HumanLikeEngine (как в сервере) ===")
try:
    from nova_chess.core import HumanLikeEngine
    eng = HumanLikeEngine(
        r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx",
        r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe",
        filter_depth=8)
    print("  создан OK, фильтр:", eng.has_filter)
    import chess
    b = chess.Board("r1bqkb1r/ppp2ppp/2n2n2/3pp1b1/3P4/2N1B2N/PPP2PPP/R2QKB1R w KQkq - 0 6")
    print("  test stockfish_moves...")
    pairs = eng.stockfish_moves(b, top=5)
    print("  OK:", [(u, round(p,3)) for u,p in pairs[:3]])
    eng.close()
except Exception as e:
    print("  ОШИБКА:", type(e).__name__, e)
    traceback.print_exc()
