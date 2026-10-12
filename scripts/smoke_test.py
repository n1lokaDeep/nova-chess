"""Быстрая проверка ядра nova_chess."""
import sys, time
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")

import chess
from nova_chess.core import NovaEngine, HumanLikeEngine, fen_to_planes, normalize_rating

print("=" * 60)
print("1. КОДИРОВАНИЕ")
print("=" * 60)
board = chess.Board()
planes = fen_to_planes(board.fen())
print(f"  форма тензора: {planes.shape}, сумма единиц: {int(planes.sum())}")
print(f"  normalize_rating(1500) = {normalize_rating(1500):.3f}")

print()
print("=" * 60)
print("2. NOVA ENGINE")
print("=" * 60)
t0 = time.time()
engine = NovaEngine(r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx")
print(f"  модель загружена за {time.time()-t0:.1f} с")

for rating in (800, 1500, 2500):
    moves = engine.predict(board, rating=rating, top=3)
    s = ", ".join(f"{m.uci}={m.percent:.0f}%" for m in moves)
    print(f"  рейтинг {rating}: {s}")

print()
print("  Проверка: сумма вероятностей всех ходов")
probs = engine.move_probabilities(board, 1500)
print(f"    ходов: {len(probs)}, сумма = {sum(probs.values()):.6f}")

print()
print("=" * 60)
print("3. СЕМПЛИРОВАНИЕ (10 раз, рейтинг 1500)")
print("=" * 60)
import collections
picks = collections.Counter()
for _ in range(10):
    m = engine.sample_move(board, rating=1500)
    picks[m.uci()] += 1
for uci, cnt in picks.most_common():
    print(f"  {uci}: {cnt}/10")

print()
print("=" * 60)
print("4. ГИБРИД С STOCKFISH")
print("=" * 60)
sf = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-unicode.exe"
import os
if not os.path.exists(sf):
    sf = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
print(f"  Stockfish: {os.path.basename(sf)}, существует: {os.path.exists(sf)}")

if os.path.exists(sf):
    hybrid = HumanLikeEngine(
        r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx", sf, filter_depth=8
    )
    print(f"  фильтр доступен: {hybrid.has_filter}")

    # Позиция, где есть очевидный зевок: белые могут взять ферзя
    tactic = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    print(f"\n  Позиция: {tactic.fen()}")
    print("  (белая ладья d1, чёрный ферзь d5 — есть взятие)")

    moves = hybrid.human_moves(tactic, rating=1500, top=4)
    print("  Чистая Nova предлагает:")
    for m in moves:
        print(f"    {m.uci}  {m.percent:.1f}%")

    print("\n  После фильтра Stockfish (strictness=0.8):")
    ev = hybrid.analyzed_moves(tactic, rating=1500, strictness=0.8)
    for e in ev:
        flag = "◀ лучший" if e.is_best else f"потеря {e.loss} сп"
        print(f"    {e.uci}  nova={e.nova_probability*100:.1f}%  оценка={e.cp}  {flag}")

    hybrid.close()

print()
print("ВСЁ РАБОТАЕТ")
