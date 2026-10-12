"""
Полная диагностика трёх багов.

1. Фильтр качества не меняет ход
2. Оценка убывает на единицу каждый ход
3. Малая разница между 800 и 2700
"""
import sys, time
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
from nova_chess.core import HumanLikeEngine

MODEL = r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx"
SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"

eng = HumanLikeEngine(MODEL, SF, filter_depth=8)

print("=" * 76)
print("БАГ 1: РАБОТАЕТ ЛИ ФИЛЬТР КАЧЕСТВА")
print("=" * 76)

# Позиция, где Nova даёт плохой ход, а Stockfish его отсеет
test = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
print(f"\nПозиция: {test.fen()}")

print("\n--- Чистая Nova (strictness = 0) ---")
for m in eng.human_moves(test, rating=800, top=6):
    print(f"  {m.uci}  {m.percent:5.1f}%")

for st in (0.0, 0.3, 0.6, 1.0):
    ev = eng.analyzed_moves(test, rating=800, strictness=st, pool=8)
    moves = "  ".join(f"{e.uci}({e.loss}cp)" for e in ev)
    print(f"\nstrictness={st:.1f}: {len(ev)} ходов -> {moves}")

print()
print("=" * 76)
print("БАГ 2: ОЦЕНКА ПОЗИЦИИ — убывает ли на единицу?")
print("=" * 76)

board = chess.Board()
print(f"\n{'ход':<4}{'чей':<8}{'оценка':>10}{'изменение':>12}")
print("-" * 36)

prev = None
for i in range(8):
    cp = eng.evaluate_position(board, depth=12)
    turn = "белые" if board.turn else "чёрные"
    delta = "" if prev is None else f"{cp - prev:+d}"
    print(f"{i+1:<4}{turn:<8}{cp:>10}{delta:>12}")
    prev = cp
    # делаем простой ход
    mv = eng.choose_move(board, rating=1500, strictness=0.5)
    if mv is None:
        break
    board.push(mv)

print()
print("=" * 76)
print("БАГ 3: РАЗНИЦА МЕЖДУ 800 И 2700")
print("=" * 76)

# Реальная разница должна проявляться в сложных позициях
complex_pos = chess.Board("r2q1rk1/pb1nbppp/1p2pn2/2p5/2BP4/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 1 12")
print(f"\nПозиция Фишера: {complex_pos.fen()}")
for r in (800, 1500, 2100, 2700):
    top = eng.human_moves(complex_pos, rating=r, top=3)
    s = "  ".join(f"{m.uci}={m.percent:.0f}%" for m in top)
    print(f"  {r:>4}: {s}")

eng.close()
print("\nГОТОВО")
