"""3.4 Komplettes Dashboard mit FastAPI.

Start:   python -m app.server            (aus dem Projektordner)
Öffnen:  http://localhost:8000  – oder von Handy/Laptop im selben WLAN:
         http://<IP-dieses-Rechners>:8000
"""
from __future__ import annotations

import asyncio
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from rcv.utils import load_config, save_config  # noqa: E402

from .db import Database  # noqa: E402
from .engine import Engine  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
db = Database()
engine = Engine(db)


@asynccontextmanager
async def lifespan(_app):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="Restaurant Analytics", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- Seiten
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/settings")
def settings_page():
    return FileResponse(STATIC / "settings.html")


# ---------------------------------------------------------------- Video
@app.get("/video")
async def video():
    """3.1 MJPEG-Stream: der Browser zeigt ihn direkt in einem <img>."""
    async def gen():
        last = None
        while True:
            jpg = engine.get_jpeg()
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot")
def snapshot():
    jpg = engine.get_jpeg(raw=True)
    if jpg is None:
        raise HTTPException(503, "Noch kein Bild verfügbar")
    return Response(jpg, media_type="image/jpeg")


# ---------------------------------------------------------------- Daten
@app.get("/api/stats")
def stats():
    """3.2 Live-Zahlen."""
    s = engine.get_stats()
    s["today"] = db.today_summary()
    return s


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


# ---------------------------------------------------------------- Einstellungen
NUMERIC = {"conf": float, "dedupe_iou": float, "dedupe_contain": float, "occupy_after_sec": float,
           "free_after_sec": float, "lock_window_sec": float, "sit_ratio": float,
           "seats_total": int, "process_width": int}


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
def set_config(new: dict):
    cfg = load_config()
    for k, cast in NUMERIC.items():
        if k in new:
            try:
                cfg[k] = cast(new[k])
            except (TypeError, ValueError):
                raise HTTPException(400, f"Ungültiger Wert für {k}")
    for k in ("source", "model", "pose_model"):
        if k in new and str(new[k]).strip():
            cfg[k] = str(new[k]).strip()
    if "tables" in new:
        tables = []
        for i, t in enumerate(new["tables"]):
            poly = t.get("polygon", [])
            if len(poly) < 3:
                raise HTTPException(400, f"Tisch {i + 1}: mindestens 3 Punkte nötig")
            tables.append({"id": str(t.get("id") or f"T{i + 1}"), "seats": int(t.get("seats", 4)),
                           "polygon": [[int(x), int(y)] for x, y in poly]})
        cfg["tables"] = tables
    save_config(cfg)
    engine.restart()
    return {"ok": True}


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn
    print(f"\n  Dashboard:  http://localhost:8000\n  Im WLAN:    http://{_local_ip()}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
