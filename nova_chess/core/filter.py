"""
Фильтр качества через Stockfish.

Nova не оценивает позиции и не считает варианты — она может предложить ход,
который человек сделал бы, но который объективно плох. Этот модуль
добавляет проверку качества: Stockfish оценивает кандидатов Nova и
отбрасывает те, что теряют слишком много.

Режим фильтрации задаётся параметром `strictness`:
    0.0 — фильтр выключен, чистая Nova (может зевнуть)
    0.5 — умеренный отсев
    1.0 — жёсткий отсев, теряется не более ~0.3 пешки
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

# Мат кодируется большим, но конечным числом, чтобы не ломать графики
MATE_SCORE = 10000
MATE_STEP = 10

# Для отображения на графике мат сжимается до этих значений (в сантипешках).
# Иначе полоса оценки улетает в бесконечность при первом же мате.
MATE_DISPLAY_CP = 1500        # ±15 пешек — визуальный максимум
BLUNDER_DISPLAY_CP = 800      # ±8 пешек — предел для обычных ошибок

# Допустимая потеря в сантипешках при strictness = 0 и 1 соответственно.
# На strictness=0 фильтр пропускает почти всё (включая зевки новичка),
# на 1 — только ходы, почти не уступающие лучшему.
#
# Верхняя граница выбрана так, чтобы на минимальном рейтинге проходили
# человеческие ошибки вплоть до потери фигуры (~900 сп), иначе модель
# играет одинаково сильно на всех уровнях.
MAX_LOSS_AT_ZERO = 900
MAX_LOSS_AT_ONE = 15


def rating_to_strictness(rating: int) -> float:
    """Переводит рейтинг игрока в строгость фильтра.

    Это ключевая калибровка: Nova сама по себе слабо реагирует на рейтинг
    (авторы применяют отдельный калибровочный слой, которого нет в
    открытом релизе). Мы воспроизводим эффект через порог качества:
    низкий рейтинг пропускает человеческие ошибки, высокий — почти нет.

    Args:
        rating: рейтинг 800–2700.

    Returns:
        Строгость 0..1 для передачи в filter().
    """
    clamped = max(800, min(2700, int(rating)))
    # На 800 — 0.0 (фильтр почти выключен: новичок зевает как новичок),
    # на 2700 — 1.0 (жёсткий отсев). Степень 0.7 растягивает нижнюю часть
    # шкалы, чтобы начальные уровни заметно отличались друг от друга.
    t = (clamped - 800) / (2700 - 800)
    return 0.85 * (t ** 0.7)


def loss_budget(strictness: float) -> int:
    """Допустимая потеря в сантипешках для заданной строгости.

    Нелинейная шкала: мягкий отсев растёт быстро, жёсткий — медленно.
    Это убирает «плато», когда 0.3 и 0.6 дают одинаковый результат.

    Args:
        strictness: 0..1.

    Returns:
        Максимальная потеря в сантипешках.
    """
    s = max(0.0, min(1.0, strictness))
    # Квадратичная интерполяция вместо линейной
    return round(MAX_LOSS_AT_ZERO + (MAX_LOSS_AT_ONE - MAX_LOSS_AT_ZERO) * (s ** 0.5))


@dataclass(frozen=True)
class EvaluatedMove:
    """Ход с оценкой Stockfish."""

    uci: str
    nova_probability: float
    centipawns: int
    loss: int

    @property
    def is_best(self) -> bool:
        """Совпадает ли ход с лучшим по Stockfish."""
        return self.loss == 0


def _score_to_cp(score: chess.engine.Score, perspective: chess.Color) -> int:
    """Переводит оценку Stockfish в сантипешки с точки зрения стороны."""
    pov = score.pov(perspective)
    if pov.is_mate():
        mate_in = pov.mate()
        # Чем быстрее мат, тем выше оценка
        magnitude = MATE_SCORE - abs(mate_in) * MATE_STEP
        return magnitude if mate_in > 0 else -magnitude
    return pov.score() or 0


def to_display_cp(cp: int, *, scale: bool = True) -> int:
    """Готовит оценку к отображению на графике.

    Сжимается ТОЛЬКО крайний случай — мат. Обычные оценки передаются
    как есть, потому что любое нелинейное преобразование искажает
    смысл: +400 (выигрыш ферзя за ладью) превращалось в +371, а
    разница между +100 и +500 становилась неразличимой.

    Args:
        cp: сырая оценка в сантипешках.
        scale: оставлен для совместимости, поведение одинаково.

    Returns:
        Оценка, пригодная для отрисовки: -1500..+1500,
        где ±1500 означает мат.
    """
    # Мат показываем как предел диапазона, а не как 10000
    if abs(cp) >= MATE_SCORE - 100:
        return MATE_DISPLAY_CP if cp > 0 else -MATE_DISPLAY_CP

    # Обычные оценки не трогаем — только предохранитель от выбросов
    return max(-MATE_DISPLAY_CP, min(MATE_DISPLAY_CP, cp))


class StockfishFilter:
    """Оценивает ходы через Stockfish и отсеивает слабые.

    Пример:
        >>> filt = StockfishFilter("stockfish.exe", depth=10)
        >>> good = filt.filter(board, candidates, strictness=0.6)
        >>> filt.close()
    """

    def __init__(self, engine_path: str | Path, *, depth: int = 10,
                 threads: int = 4, hash_mb: int = 256) -> None:
        """Запускает процесс Stockfish.

        Args:
            engine_path: путь к исполняемому файлу Stockfish.
            depth: глубина анализа для фильтрации (меньше = быстрее).
            threads: число потоков Stockfish.
            hash_mb: размер хеш-таблицы в мегабайтах.
        """
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Stockfish не найден: {engine_path}")

        self._engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        self._engine.configure({"Threads": threads, "Hash": hash_mb})
        self._lock = threading.Lock()
        self.depth = depth

    def evaluate_move(self, board: chess.Board, uci: str, depth: int | None = None) -> int:
        """Оценка позиции после хода, с точки зрения сделавшего ход.

        Returns:
            Оценка в сантипешках.
        """
        move = chess.Move.from_uci(uci)
        after = board.copy()
        after.push(move)

        with self._lock:
            info = self._engine.analyse(
                after, chess.engine.Limit(depth=depth or self.depth)
            )

        # Оценка даётся с точки зрения того, чей ход ПОСЛЕ —
        # значит для сделавшего ход её надо инвертировать
        return -_score_to_cp(info["score"], after.turn)

    def evaluate_all(
        self, board: chess.Board, candidates: list[tuple[str, float]],
        depth: int | None = None,
    ) -> list[EvaluatedMove]:
        """Оценивает список кандидатов и считает потерю относительно лучшего.

        Ошибки различаются по смыслу: некорректный ход просто пропускается,
        а падение движка поднимается наверх. Раньше любая ошибка молча
        съедалась, и вызывающий код не мог отличить «нет ходов» от
        «Stockfish умер».

        Raises:
            RuntimeError: если движок перестал отвечать.
        """
        scored: list[tuple[str, float, int]] = []

        for uci, probability in candidates:
            try:
                cp = self.evaluate_move(board, uci, depth)
            except ValueError:
                # ход не разбирается — это вопрос данных, не движка
                continue
            except chess.engine.EngineError as exc:
                raise RuntimeError(
                    f"Stockfish не отвечает: {exc}. Возможно, процесс упал."
                ) from exc
            scored.append((uci, probability, cp))

        if not scored:
            return []

        best_cp = max(cp for _, _, cp in scored)
        return [
            EvaluatedMove(uci, prob, cp, best_cp - cp)
            for uci, prob, cp in scored
        ]

    def filter(
        self,
        board: chess.Board,
        candidates: list[tuple[str, float]],
        *,
        strictness: float = 0.6,
        depth: int | None = None,
    ) -> list[EvaluatedMove]:
        """Отсеивает ходы, теряющие слишком много относительно лучшего.

        Args:
            candidates: список (uci, вероятность Nova).
            strictness: 0 = без фильтрации, 1 = жёсткий отсев.

        Returns:
            Отфильтрованный список EvaluatedMove.
            Если после отсева не осталось ходов — возвращает лучший по Stockfish.
        """
        strictness = max(0.0, min(1.0, strictness))
        evaluated = self.evaluate_all(board, candidates, depth)
        if not evaluated:
            return []

        if strictness <= 0.01:
            return evaluated

        allowed_loss = loss_budget(strictness)
        kept = [e for e in evaluated if e.loss <= allowed_loss]

        if kept:
            return kept
        # Всё отсеялось — отдаём объективно лучший
        return [min(evaluated, key=lambda e: e.loss)]

    def close(self) -> None:
        """Останавливает процесс Stockfish."""
        # движок мог уже умереть — гасить повторно не страшно
        with self._lock, contextlib.suppress(Exception):
            self._engine.quit()

    def __enter__(self) -> StockfishFilter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
