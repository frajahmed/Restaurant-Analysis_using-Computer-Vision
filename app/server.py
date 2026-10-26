"""Step 3.4 – Full dashboard (FastAPI).

Start:  ``python main.py``  (or ``python -m app.server``)
Open:   http://localhost:8000 – or from a phone/laptop on the same Wi-Fi:
        http://<IP of this computer>:8000
"""
from __future__ import annotations

import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from rcv.config import Config, ConfigError  # noqa: E402
from rcv.streaming import MJPEG_MEDIA_TYPE  # noqa: E402

from .db import Database  # noqa: E402
from .engine import Engine  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Listen on all interfaces so phones/laptops on the same Wi-Fi (or Docker) can connect.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


def create_app(engine: Engine, db: Database) -> FastAPI:
    """Build the FastAPI application. The engine starts and stops with the server."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="Restaurant Analytics", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def dashboard_page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/settings", include_in_schema=False)
    def settings_page():
        return FileResponse(STATIC_DIR / "settings.html")

    # ---------------------------------------------------------------- video
    @app.get("/video")
    def video():
        """Live MJPEG stream of the annotated frames."""
        return StreamingResponse(engine.annotated.mjpeg_stream(), media_type=MJPEG_MEDIA_TYPE)

    @app.get("/api/snapshot")
    def snapshot():
        """Current frame without drawings (used by the table editor)."""
        jpeg = engine.raw.get()
        if jpeg is None:
            raise HTTPException(503, "Noch kein Bild verfügbar")
        return Response(jpeg, media_type="image/jpeg")

    # ---------------------------------------------------------------- data
    @app.get("/api/stats")
    def stats():
        """Live key figures plus today's totals."""
        return {**engine.stats, "today": db.today_summary()}

    @app.get("/api/visits")
    def visits(limit: int = 50):
        return db.recent_visits(limit)

    @app.get("/api/history")
    def history():
        return db.hourly()

    @app.delete("/api/history")
    def clear_history():
        db.clear()
        return {"ok": True}

    # ---------------------------------------------------------------- settings
    @app.get("/api/config")
    def get_config():
        return Config.load().to_dict()

    @app.post("/api/config")
    def set_config(changes: dict):
        """Validate and save new settings, then restart the analysis."""
        config = Config.load()
        try:
            config.update(changes)
        except (ConfigError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        config.save()
        engine.restart()
        return {"ok": True}

    return app


def local_ip() -> str:
    """IP address of this computer in the local network (for the start message)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))     # no data is sent; only picks the right interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# Module-level app so that `uvicorn app.server:app` also works.
database = Database()
app = create_app(Engine(database), database)


def main() -> None:
    import uvicorn

    print(f"\n  Dashboard:  http://localhost:{PORT}\n  Im WLAN:    http://{local_ip()}:{PORT}\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
