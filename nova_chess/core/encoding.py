"""
Кодирование шахматной позиции в тензор для модели Nova.

Модель принимает:
  positions     — float32 (B, 18, 8, 8), 18 плоскостей
  conditioning  — float32 (B, 3): (rating_norm, classical, aggression)

Плоскости доски:
  0-5   — белые фигуры: пешка, конь, слон, ладья, ферзь, король
  6-11  — чёрные фигуры в том же порядке
  12    — чей ход (1 = белые)
  13-16 — права рокировки: K, Q, k, q
  17    — поле взятия на проходе
"""

from __future__ import annotations

import numpy as np

# Соответствие символа FEN индексу плоскости
PIECE_TO_PLANE: dict[str, int] = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

MIN_RATING = 800
MAX_RATING = 2700


def fen_to_planes(fen: str) -> np.ndarray:
    """Преобразует FEN в тензор плоскостей (18, 8, 8).

    Args:
        fen: строка позиции в формате FEN.

    Returns:
        Массив float32 формы (18, 8, 8).

    Raises:
        ValueError: если FEN некорректен.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    parts = fen.split()

    if len(parts) < 4:
        raise ValueError(f"Некорректный FEN: нужно минимум 4 поля, получено {len(parts)}")

    board_part, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]

    for rank_index, rank_str in enumerate(board_part.split("/")):
        # FEN читается сверху вниз (8-й ряд первым), массив — снизу вверх
        row = 7 - rank_index
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
            else:
                plane = PIECE_TO_PLANE.get(ch)
                if plane is None:
                    raise ValueError(f"Неизвестная фигура в FEN: {ch!r}")
                if col > 7:
                    raise ValueError(f"Переполнение ряда в FEN: {rank_str!r}")
                planes[plane, row, col] = 1.0
                col += 1

    if turn == "w":
        planes[12].fill(1.0)

    for char, plane in (("K", 13), ("Q", 14), ("k", 15), ("q", 16)):
        if char in castling:
            planes[plane].fill(1.0)

    if ep != "-" and len(ep) == 2:
        file_index = ord(ep[0]) - ord("a")
        if 0 <= file_index < 8:
            planes[17, 0, file_index] = 1.0

    return planes


def normalize_rating(rating: float) -> float:
    """Нормализует рейтинг в диапазон [0, 1] так, как это делает Nova.

    Модель обучена на диапазоне 800–2700; значения вне его обрезаются.
    """
    clamped = max(MIN_RATING, min(MAX_RATING, int(rating)))
    return (clamped - MIN_RATING) / (MAX_RATING - MIN_RATING)


def make_conditioning(rating: int, classical: float, aggression: float) -> np.ndarray:
    """Собирает вектор условий (1, 3) для модели.

    Args:
        rating: рейтинг игрока, 800–2700.
        classical: 0 = гипермодернизм (Nf3, c4), 1 = классика (e4, d4).
        aggression: 0 = спокойная игра, 1 = тактика и атака.

    Returns:
        Массив float32 формы (1, 3).
    """
    return np.array(
        [[normalize_rating(rating), float(classical), float(aggression)]],
        dtype=np.float32,
    )
