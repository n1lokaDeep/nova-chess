import sys
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
import chess

# Восстанавливаю позицию со скриншота
# Белые: Ra1 Nb1 Bc1 Qd1 Ke1 Rh1, пешки a2 b2 c2 d2 e2 f2 g2 h2, Be3, Nh3, d4
# Чёрные: Ra8 Be8(?) Qd8 Ke8 Rg8, Bc8, пешки a7 b7 c7 d7 f7 g7 h7, Nf6, e5(?) Bg5
fen = "r1bqkb1r/ppp2ppp/2n2n2/3pp1b1/3P4/2N1B2N/PPP2PPP/R2QKB1R w KQkq - 0 6"
try:
    b = chess.Board(fen)
    print("Тестовый FEN валиден:", b.fen())
    print("Ход:", "белые" if b.turn else "чёрные")
    print("Легальных ходов:", b.legal_moves.count())
    print()
    print("Ходы, начиная с f7 или g6:", [m.uci() for m in b.legal_moves if m.uci()[:2] in ("f7","g6")])
except Exception as e:
    print("FEN невалиден:", e)

print()
print("=" * 60)
print("ЧИТАЮ FEN ПРЯМО С ДОСКИ ЧЕРЕЗ ПОЛЕ ВВОДА")
print("=" * 60)
print("Нужен реальный FEN со скриншота — расставляю по фигурам:")
print()
print("Белые вижу: Ke1, Qd1, Ra1, Rh1, Nb1, Bc1, Be3, Nh3")
print("            пешки: a2 b2 c2 d2 e2 f2 g2 h2, пешка d4")
print("Чёрные вижу: Ke8, Qd8, Ra8, Rg8, Nb8(?), Bc8, Be8(?), Bg5")
print("             пешки: a7 b7 c7 d7 f7 g7 h7, пешка e5")
