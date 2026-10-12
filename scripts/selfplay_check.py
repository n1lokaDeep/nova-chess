import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess, chess.engine
from nova_chess.core import HumanLikeEngine
from nova_chess.core.filter import _score_to_cp

SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
eng = HumanLikeEngine(r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx", SF, filter_depth=8)
ref = chess.engine.SimpleEngine.popen_uci(SF)
ref.configure({"Threads": 4, "Hash": 256})

print("=" * 70)
print("ТЕСТ: САМОИГРА через мой stockfish_moves")
print("Если оценка рушится — баг остался")
print("=" * 70)
print()

b = chess.Board()
print(f"{'#':<4}{'чей':<7}{'мой ход':<9}{'эталон SF':<11}{'оценка':>9}")
print("-" * 42)
bad = 0
for i in range(20):
    if b.is_game_over(): break
    mover = "бел" if b.turn else "чёр"

    # что предлагает мой код
    pairs = eng.stockfish_moves(b, top=3, depth=12)
    mine = pairs[0][0] if pairs else None

    # что говорит Stockfish напрямую
    best = ref.play(b, chess.engine.Limit(depth=14)).move.uci()

    info = ref.analyse(b, chess.engine.Limit(depth=14))
    cp = _score_to_cp(info["score"], chess.WHITE)

    flag = ""
    if mine != best:
        bad += 1
        flag = "  <-- РАСХОЖДЕНИЕ"
    print(f"{i+1:<4}{mover:<7}{str(mine):<9}{best:<11}{cp:>9}{flag}")

    if mine is None: break
    b.push(chess.Move.from_uci(mine))

print()
print(f"Расхождений с эталоном: {bad} из 20")
print("ОЦЕНКА:", _score_to_cp(ref.analyse(b, chess.engine.Limit(depth=14))["score"], chess.WHITE))
ref.quit(); eng.close()
