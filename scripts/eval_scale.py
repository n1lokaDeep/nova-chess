import sys, math
sys.path.insert(0, r"C:\Users\Orion\Projects\nova-chess")
from nova_chess.core.filter import to_display_cp, BLUNDER_DISPLAY_CP, MATE_DISPLAY_CP

print("=== ЧТО ДЕЛАЕТ СЖАТИЕ tanh ===")
print(f"BLUNDER_DISPLAY_CP = {BLUNDER_DISPLAY_CP}")
print()
print(f"{'сырое':>8}{'после сжатия':>16}")
for cp in (-10000, -2000, -1000, -500, -402, -100, 0, 100, 402, 500, 1000, 2000, 10000):
    print(f"{cp:>8}{to_display_cp(cp):>16}")

print()
print("=== ПРОБЛЕМА ===")
print("tanh сжимает ВСЁ. Оценка +402 становится:", to_display_cp(402))
print("Разница между +100 и +500 почти не видна — график слепой.")
