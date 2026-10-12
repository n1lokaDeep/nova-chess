"""
Тест висящих фигур. Позиции строятся ХОДАМИ от начальной,
чтобы гарантировать валидность — вручную написанные FEN давали
невозможные позиции, на которых Stockfish падает.
"""
import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess
import chess.engine
from nova_chess.core import HumanLikeEngine
from nova_chess.core.filter import _score_to_cp

SF = r"C:\Users\Orion\Chess\sf\stockfish\stockfish-windows-x86-64-universal.exe"
MODEL = r"C:\Users\Orion\Projects\nova-chess\models\nova_v3b.onnx"


def from_moves(moves):
    """Строит позицию из последовательности ходов и проверяет валидность."""
    b = chess.Board()
    for m in moves:
        b.push(chess.Move.from_uci(m))
    status = b.status()
    if status != chess.STATUS_VALID:
        raise ValueError(f"позиция невалидна (код {status})")
    return b


CASES = {
    "Стартовая позиция": [],
    "После 1.e4, ход чёрных": ["e2e4"],
    "После 1.e4 e5 2.Nf3 Nc6, ход белых":
        ["e2e4", "e7e5", "g1f3", "b8c6"],
    "После 1.e4 e5 2.Nf3 Nc6 3.Bb5, ход чёрных":
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    "Итальянская, ход чёрных":
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
}

SPECIAL = {
    "Ферзь под боем ладьи (ход белых)": "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
    "Ферзь под боем ладьи (ход чёрных)": "3Rk3/8/8/3q4/8/8/8/4K3 b - - 0 1",
}

print("=" * 74)
print("ПРОВЕРКА: ходы Stockfish ранжируются ЗА ТОГО, КТО ХОДИТ")
print("=" * 74)

eng = HumanLikeEngine(MODEL, SF, filter_depth=8)
ref = chess.engine.SimpleEngine.popen_uci(SF)
ref.configure({"Threads": 4, "Hash": 256})

ok_count = 0
fail_count = 0


def check(label, board):
    """Сравнивает мой топ-ход с эталонным Stockfish."""
    global ok_count, fail_count
    my = eng.stockfish_moves(board, top=3, depth=14)
    best = ref.play(board, chess.engine.Limit(depth=16)).move.uci()
    cp = _score_to_cp(
        ref.analyse(board, chess.engine.Limit(depth=14))["score"], chess.WHITE)

    match = bool(my) and my[0][0] == best
    if match:
        ok_count += 1
    else:
        fail_count += 1

    turn = "белые" if board.turn else "чёрные"
    print(f"\n{label}")
    print(f"  ход: {turn}   оценка: {cp:+d}")
    print(f"  Stockfish: {best}")
    print(f"  мой топ:   {my[0][0] if my else '-'}  " +
          ("OK" if match else "РАСХОЖДЕНИЕ"))
    if my:
        print(f"  далее:     {[u for u, _ in my[1:]]}")


try:
    for label, moves in CASES.items():
        try:
            board = from_moves(moves)
        except ValueError as exc:
            print(f"\n{label}: ПРОПУСК — {exc}")
            continue
        check(label, board)

    for label, fen in SPECIAL.items():
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            print(f"\n{label}: ПРОПУСК — {exc}")
            continue
        if board.status() != chess.STATUS_VALID:
            print(f"\n{label}: ПРОПУСК — невалидная позиция ({board.status()})")
            continue
        check(label, board)

finally:
    print()
    print("=" * 74)
    print(f"ИТОГ: совпало {ok_count}, расхождений {fail_count}")
    print("=" * 74)
    ref.quit()
    eng.close()
