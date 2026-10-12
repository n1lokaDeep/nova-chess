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

from ..core import HumanLikeEngine

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "nova_v3b.onnx"
UI_FILE = HERE / "ui.html"

PORT = int(os.environ.get("NOVA_PORT", "8765"))
STOCKFISH = os.environ.get("NOVA_STOCKFISH_PATH")

_engine: HumanLikeEngine | None = None
_lock = threading.Lock()


def get_engine() -> HumanLikeEngine:
    """Ленивая инициализация движка."""
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
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
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

        if path == "/api/state":
            board = chess.Board(data["fen"])
            payload = {
                "legal": [m.uci() for m in board.legal_moves],
                "turn": "w" if board.turn else "b",
                "check": board.is_check(),
                "over": board.is_game_over(),
                "reason": board.result() if board.is_game_over() else "",
                "moves": [
                    [mv.uci, mv.probability]
                    for mv in engine.human_moves(
                        board,
                        data["rating"], data["classical"], data["aggression"],
                        top=6,
                    )
                ],
            }
            if engine.has_filter:
                payload["eval"] = engine.evaluate_position(board, depth=10)
            self._send(200, json.dumps(payload).encode())

        elif path == "/api/analyzed":
            board = chess.Board(data["fen"])
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
            board = chess.Board(data["fen"])
            move = chess.Move.from_uci(data["move"])
            if move not in board.legal_moves:
                self._send(200, json.dumps({"error": "нелегальный ход"}).encode())
                return
            board.push(move)
            self._send(200, json.dumps({"fen": board.fen()}).encode())

        elif path == "/api/nova":
            board = chess.Board(data["fen"])
            move = engine.choose_move(
                board,
                data["rating"], data["classical"], data["aggression"],
                strictness=data.get("strictness", 0.6),
                sharpness=data.get("sharpness", 3.0),
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
