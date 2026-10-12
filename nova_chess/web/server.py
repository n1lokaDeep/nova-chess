"""
Веб-сервер Nova Chess.

Запуск:
    python -m nova_chess.web.server

Затем откройте http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess

from ..core import HumanLikeEngine, MaiaEngine

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "nova_v3b.onnx"
UI_FILE = HERE / "ui.html"

PORT = int(os.environ.get("NOVA_PORT", "8765"))
STOCKFISH = os.environ.get("NOVA_STOCKFISH_PATH")

_engine: HumanLikeEngine | None = None
_maia: MaiaEngine | None = None
_maia_failed = False
_lock = threading.Lock()

# Единая очередь для ВСЕХ обращений к Stockfish.
# ThreadingHTTPServer создаёт поток на каждый запрос, а процесс Stockfish
# один: без внешней блокировки потоки дерутся за него, event loop
# python-chess ломается и движок умирает с «engine event loop dead».
_engine_busy = threading.Lock()


def _describe_status(status: int) -> str:
    """Переводит битовую маску chess.STATUS_* в понятное объяснение."""
    reasons = [
        (chess.STATUS_NO_WHITE_KING, "нет белого короля"),
        (chess.STATUS_NO_BLACK_KING, "нет чёрного короля"),
        (chess.STATUS_TOO_MANY_KINGS, "больше двух королей"),
        (chess.STATUS_TOO_MANY_WHITE_PAWNS, "слишком много белых пешек"),
        (chess.STATUS_TOO_MANY_BLACK_PAWNS, "слишком много чёрных пешек"),
        (chess.STATUS_PAWNS_ON_BACKRANK, "пешка на последней горизонтали"),
        (chess.STATUS_TOO_MANY_WHITE_PIECES, "слишком много белых фигур"),
        (chess.STATUS_TOO_MANY_BLACK_PIECES, "слишком много чёрных фигур"),
        (chess.STATUS_BAD_CASTLING_RIGHTS, "неверные права рокировки"),
        (chess.STATUS_INVALID_EP_SQUARE, "неверное поле взятия на проходе"),
        (chess.STATUS_OPPOSITE_CHECK, "король под боем у стороны, которая не ходит"),
        (chess.STATUS_IMPOSSIBLE_CHECK, "невозможный шах"),
    ]
    found = [text for flag, text in reasons if status & flag]
    return ", ".join(found) if found else f"код {status}"


def _safe_board(fen: str) -> chess.Board:
    """Создаёт позицию и проверяет, что её можно отдать движку.

    Raises:
        ValueError: если FEN некорректен или позиция невозможна.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"некорректный FEN: {exc}") from exc

    status = board.status()
    if status != chess.STATUS_VALID:
        raise ValueError(f"позиция невозможна: {_describe_status(status)}")
    return board


def get_engine() -> HumanLikeEngine:
    """Ленивая инициализация движка Nova + Stockfish."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = HumanLikeEngine(
                    DEFAULT_MODEL,
                    STOCKFISH if STOCKFISH and Path(STOCKFISH).exists() else None,
                    filter_depth=8,
                )
    return _engine


def restart_engine() -> None:
    """Перезапускает движок, если процесс Stockfish умер.

    python-chess бросает EngineTerminatedError, когда процесс упал;
    после этого движок бесполезен, и его нужно поднять заново.
    """
    global _engine
    with _lock:
        if _engine is not None:
            try:
                _engine.close()
            except Exception:
                pass
        _engine = HumanLikeEngine(
            DEFAULT_MODEL,
            STOCKFISH if STOCKFISH and Path(STOCKFISH).exists() else None,
            filter_depth=8,
        )


def engine_alive() -> bool:
    """Проверяет, жив ли процесс Stockfish.

    Намеренно НЕ использует ping(): он добавляет лишний обмен с движком
    и сам может спровоцировать падение. Достаточно проверить, что
    процесс существует.
    """
    if _engine is None or _engine._filter is None:
        return True
    try:
        proc = getattr(_engine._filter._engine, "protocol", None)
        if proc is None:
            return True
        transport = getattr(proc, "_transport", None)
        if transport is None:
            return True
        return not transport.is_closing()
    except Exception:
        # Не смогли определить — считаем живым, ошибка всплывёт при запросе
        return True


def get_maia() -> MaiaEngine | None:
    """Ленивая инициализация Maia-2.

    Возвращает None, если пакет не установлен — интерфейс продолжит
    работать с Nova и Stockfish.
    """
    global _maia, _maia_failed
    if _maia is None and not _maia_failed:
        with _lock:
            if _maia is None and not _maia_failed:
                try:
                    _maia = MaiaEngine()
                except Exception:
                    _maia_failed = True
    return _maia


class Handler(BaseHTTPRequestHandler):
    """HTTP-обработчик для доски."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: D102
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            if UI_FILE.exists():
                # charset=utf-8 обязателен: файл в UTF-8 без BOM,
                # иначе браузер читает кириллицу как cp1251
                self._send(200, UI_FILE.read_bytes(),
                           "text/html; charset=utf-8")
            else:
                self._send(500, b'{"error":"ui.html not found"}')
        elif self.path == "/api/health":
            engine = get_engine()
            self._send(200, json.dumps({
                "ok": True,
                "filter": engine.has_filter,
            }).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "bad json"}).encode())
            return

        try:
            self._route(data)
        except Exception as exc:  # noqa: BLE001
            self._send(200, json.dumps({"error": str(exc)}).encode())

    def _route(self, data: dict) -> None:
        engine = get_engine()
        path = self.path

        # Все обращения к Stockfish идут через одну очередь: параллельные
        # запросы ломают event loop python-chess и убивают процесс движка
        with _engine_busy:
            try:
                if not engine_alive():
                    restart_engine()
                engine = get_engine()
                self._dispatch(engine, path, data)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                # Смерть движка лечится перезапуском, а не отдачей мусора
                if "event loop" in msg or "Terminated" in msg or "died" in msg:
                    try:
                        restart_engine()
                        self._dispatch(get_engine(), path, data)
                        return
                    except Exception as retry_exc:  # noqa: BLE001
                        msg = f"движок перезапущен, но снова упал: {retry_exc}"
                self._send(200, json.dumps({"error": msg}).encode())

    def _dispatch(self, engine: HumanLikeEngine, path: str, data: dict) -> None:
        if path == "/api/validate":
            raw = (data.get("fen") or "").strip()
            if not raw:
                self._send(200, json.dumps(
                    {"ok": False, "error": "пустой FEN"}).encode())
                return
            try:
                board = _safe_board(raw)
            except ValueError as exc:
                self._send(200, json.dumps(
                    {"ok": False, "error": str(exc)}).encode())
                return

            # Stockfish 19 падает на позициях, которые python-chess
            # формально разбирает, но которые невозможны (лишние фигуры,
            # король под боем и т.п.). Проверяем статус и не пускаем
            # такие позиции в движок — иначе процесс умирает.
            status = board.status()
            if status != chess.STATUS_VALID:
                reason = _describe_status(status)
                self._send(200, json.dumps({
                    "ok": False,
                    "error": f"позиция невозможна: {reason}",
                }).encode())
                return

            self._send(200, json.dumps({
                "ok": True,
                "fen": board.fen(),                      # нормализованный
                "legalCount": board.legal_moves.count(),
                "turn": "w" if board.turn else "b",
                "check": board.is_check(),
                "over": board.is_game_over(),
            }).encode())

        if path == "/api/state":
            board = _safe_board(data["fen"])
            source = data.get("source", "nova")

            # strictness = None означает "авто": порог выводится из рейтинга
            raw_strict = data.get("strictness")
            strictness = None if raw_strict is None else float(raw_strict)

            want_eval = bool(data.get("wantEval", True))

            if source == "sf" and engine.has_filter:
                # Чистый Stockfish: играет объективно лучшие ходы.
                sf_depth = data.get("sfDepth")
                pairs = engine.stockfish_moves(
                    board, top=5,
                    depth=int(sf_depth) if sf_depth else None,
                )
                moves = [[uci, prob] for uci, prob in pairs]

            elif source == "maia":
                maia = get_maia()
                if maia is None:
                    self._send(200, json.dumps({
                        "error": "Maia-2 не установлена (pip install maia2)"
                    }).encode())
                    return
                rating = int(data["rating"])
                top = maia.predict(
                    board, elo_self=rating,
                    elo_oppo=int(data.get("oppoRating", rating)),
                    top=6,
                )
                moves = [[m.uci, m.probability] for m in top]

            elif (strictness is None or strictness > 0.01) and engine.has_filter:
                analyzed = engine.analyzed_moves(
                    board,
                    data["rating"], data["classical"], data["aggression"],
                    strictness=strictness, pool=8,
                )
                moves = [[m.uci, m.nova_probability] for m in analyzed]

            else:
                moves = [
                    [mv.uci, mv.probability]
                    for mv in engine.human_moves(
                        board,
                        data["rating"], data["classical"], data["aggression"],
                        top=6,
                    )
                ]

            payload = {
                "legal": [m.uci() for m in board.legal_moves],
                "turn": "w" if board.turn else "b",
                "check": board.is_check(),
                "over": board.is_game_over(),
                "reason": board.result() if board.is_game_over() else "",
                "source": source,
                "strictness": strictness,
                "moves": moves,
            }
            # Оценку считаем отдельно: она дорогая, а интерфейсу нужна
            # не всегда (например, при быстром клике по доске)
            if engine.has_filter and want_eval:
                payload["eval"] = engine.evaluate_for_display(board, depth=12)
            self._send(200, json.dumps(payload).encode())

        elif path == "/api/analyzed":
            board = _safe_board(data["fen"])
            moves = engine.analyzed_moves(
                board,
                data["rating"], data["classical"], data["aggression"],
                strictness=data.get("strictness", 0.6),
            )
            self._send(200, json.dumps({
                "moves": [
                    {
                        "uci": m.uci,
                        "probability": m.nova_probability,
                        "centipawns": m.centipawns,
                        "loss": m.loss,
                    }
                    for m in moves
                ]
            }).encode())

        elif path == "/api/play":
            board = _safe_board(data["fen"])
            move = chess.Move.from_uci(data["move"])
            if move not in board.legal_moves:
                self._send(200, json.dumps({"error": "нелегальный ход"}).encode())
                return
            board.push(move)
            self._send(200, json.dumps({"fen": board.fen()}).encode())

        elif path == "/api/nova":
            board = _safe_board(data["fen"])
            source = data.get("source", "nova")

            if source == "sf" and engine.has_filter:
                # Stockfish: берём его лучший ход
                sf_depth = data.get("sfDepth")
                pairs = engine.stockfish_moves(
                    board, top=1,
                    depth=int(sf_depth) if sf_depth else None,
                )
                move = chess.Move.from_uci(pairs[0][0]) if pairs else None

            elif source == "maia":
                maia = get_maia()
                if maia is None:
                    self._send(200, json.dumps({
                        "error": "Maia-2 не установлена (pip install maia2)"
                    }).encode())
                    return
                rating = int(data["rating"])
                # Maia умеет семплировать — используем её выбор,
                # чтобы партия не была каждый раз одинаковой
                move = maia.sample_move(
                    board, elo_self=rating,
                    elo_oppo=int(data.get("oppoRating", rating)),
                    sharpness=float(data.get("sharpness", 1.0)),
                )

            else:
                move = engine.choose_move(
                    board,
                    data["rating"], data["classical"], data["aggression"],
                    strictness=data.get("strictness"),
                    sharpness=float(data.get("sharpness", 1.0)),
                )

            if move is None:
                self._send(200, json.dumps({"error": "нет ходов"}).encode())
                return
            board.push(move)
            self._send(200, json.dumps({
                "move": move.uci(), "fen": board.fen()
            }).encode())

        elif path == "/api/replay":
            board = chess.Board()
            for uci in data.get("history", []):
                board.push(chess.Move.from_uci(uci))
            self._send(200, json.dumps({"fen": board.fen()}).encode())

        else:
            self._send(404, b"{}")


def main() -> None:
    """Запускает сервер и открывает браузер."""
    if not DEFAULT_MODEL.exists():
        raise SystemExit(
            f"Модель не найдена: {DEFAULT_MODEL}\n"
            "Скачайте nova_v3b.onnx с Hugging Face (см. README)."
        )

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Nova Chess запущен: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено")
        if _engine is not None:
            _engine.close()


if __name__ == "__main__":
    main()
