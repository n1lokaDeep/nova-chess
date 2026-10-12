# Nova Chess

**Шахматный движок, который играет как человек — и не зевает.**

[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Гибрид двух моделей: **Nova** предсказывает ходы, которые делают живые люди
заданного рейтинга, а **Stockfish** отсеивает объективно слабые ходы.

---

## Проблема

Обычные движки ослабляют **случайными зевками**. Stockfish на «уровне 1200»
выдаёт гениальные ходы вперемешку с идиотскими — так люди не играют.

**Nova** обучалась на **500 миллионах партий Lichess** и предсказывает
именно те ходы, которые выбирают люди. Но у неё нет оценки позиции:
она не накажет вас за грубый зевок.

**Решение:** Nova предлагает человеческие ходы → Stockfish проверяет качество
→ движок выбирает из тех, что и по-человечески, и не глупо.

---

## Возможности

| Функция | Описание |
|---|---|
| 🎭 **Человеческие ходы** | распределение вероятностей по всем легальным ходам |
| 📊 **Настройка рейтинга** | 800–2700, отдельная модель поведения для каждого уровня |
| 🎨 **Две оси стиля** | классика ↔ гипермодернизм, спокойствие ↔ агрессия |
| 🛡️ **Фильтр качества** | Stockfish отсеивает ходы, теряющие больше N сантипешек |
| ♟️ **CLI** | четыре команды: `moves`, `play`, `compare`, `selfplay` |
| 🌐 **Веб-интерфейс** | доска с ползунками рейтинга и стиля |
| ✅ **47 тестов** | покрытие кодирования, движка, фильтра, гибрида |

---

## Установка

```bash
git clone https://github.com/USER/nova-chess.git
cd nova-chess
pip install -r requirements.txt
```

### Модели

**Nova** (381 МБ) — скачайте с Hugging Face:

```bash
# файл конфигурации
curl -L -o models/nova_v3b.onnx \
  https://huggingface.co/novachess/novachess-engine/resolve/main/nova_v3b.onnx

# веса (380 МБ)
curl -L -o models/nova_v3b.onnx.data \
  https://huggingface.co/novachess/novachess-engine/resolve/main/nova_v3b.onnx.data
```

**Stockfish** — с [stockfishchess.org/download](https://stockfishchess.org/download/),
положите путь в переменную окружения:

```bash
export NOVA_STOCKFISH_PATH=/path/to/stockfish
```

---

## Использование

### CLI

```bash
# Какие ходы сделал бы человек рейтинга 1500?
python -m nova_chess.cli moves --rating 1500 --top 5

# Один ход движка с фильтром качества
python -m nova_chess.cli play --rating 1800 --strictness 0.7

# Сравнить чистую Nova и фильтрованную
python -m nova_chess.cli compare --stockfish /path/to/stockfish

# Партия Nova против Nova
python -m nova_chess.cli selfplay --moves 40 --rating 2000
```

Пример вывода:

```
$ python -m nova_chess.cli moves --rating 1500 --top 4
Позиция: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Ход белые
Рейтинг: 1500   Стиль: 0.50   Агрессия: 0.50

Ход      Вероятность
------------------------
e2e4            55.1%
d2d4            24.7%
c2c4             5.6%
g1f3             3.7%
```

### Python API

```python
import chess
from nova_chess.core import HumanLikeEngine

engine = HumanLikeEngine(
    "models/nova_v3b.onnx",
    "stockfish/stockfish.exe",
)

board = chess.Board()

# Что сыграл бы человек 1500?
for mv in engine.human_moves(board, rating=1500, top=3):
    print(f"{mv.uci}  {mv.percent:.1f}%")

# Ход с контролем качества
move = engine.choose_move(board, rating=1500, strictness=0.7)
print(move.uci())

engine.close()
```

### Только Nova, без Stockfish

```python
from nova_chess.core import NovaEngine

nova = NovaEngine("models/nova_v3b.onnx")

# Вероятности всех легальных ходов
probs = nova.move_probabilities(chess.Board(), rating=1500)

# Сэмплировать ход как человек
move = nova.sample_move(chess.Board(), rating=1500, sharpness=3.0)
```

---

## Как это работает

### Кодирование позиции

Модель принимает тензор `(18, 8, 8)`:

| Плоскости | Содержимое |
|---|---|
| 0–5 | белые: пешка, конь, слон, ладья, ферзь, король |
| 6–11 | чёрные в том же порядке |
| 12 | чей ход |
| 13–16 | права рокировки: K, Q, k, q |
| 17 | поле взятия на проходе |

Плюс вектор условий `(rating_norm, classical, aggression)`.

### Пространство ходов

`64 × 64 = 4096` комбинаций «откуда-куда» плюс три блока превращений
(конь, слон, ладья) — итого **16384 индекса**. Превращение в ферзя
использует смещение 0 и восстанавливается по позиции.

### Фильтр качества

```
strictness = 0.0  →  чистая Nova (может зевнуть)
strictness = 0.6  →  умеренный отсев
strictness = 1.0  →  потеря не более 30 сантипешек
```

Допустимая потеря интерполируется линейно:

```python
allowed = 400 + (30 - 400) * strictness   # в сантипешках
```

---

## Структура проекта

```
nova-chess/
├── nova_chess/
│   ├── core/
│   │   ├── encoding.py    # FEN → тензор, нормализация рейтинга
│   │   ├── engine.py      # обёртка над ONNX-моделью Nova
│   │   ├── filter.py      # фильтр качества через Stockfish
│   │   └── hybrid.py      # объединение Nova + Stockfish
│   ├── web/               # веб-интерфейс
│   └── cli.py             # командная строка
├── models/                # nova_v3b.onnx (не в git)
├── tests/                 # 47 тестов
├── scripts/               # вспомогательные скрипты
└── docs/                  # документация и скриншоты
```

---

## Тесты

```bash
pytest tests/ -v
```

```
47 passed in 10.79s
```

Тесты, требующие модель или Stockfish, автоматически пропускаются,
если файлы не найдены.

---

## Ограничения

- **Nova не считает варианты.** Это policy-сеть: один проход, без поиска.
  Она не находит тактику в глубину.
- **Рейтинг от 800.** Ниже модель не обучена.
- **Фильтр замедляет ход.** Каждый кандидат оценивается Stockfish —
  около 0.1 с на позицию при глубине 8.
- **Нет графиков оценки партии.** Nova не даёт оценки позиции.

---

## Благодарности

- **[Nova](https://novachess.ai)** — модель человеческих ходов
- **[Maia](https://maiachess.com)** — пионер направления human-like движков
- **[python-chess](https://python-chess.readthedocs.io)** — работа с шахматами
- **[Stockfish](https://stockfishchess.org)** — эталонный движок

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Модель Nova распространяется отдельно её авторами;
веса скачиваются с Hugging Face и в репозиторий не входят.
