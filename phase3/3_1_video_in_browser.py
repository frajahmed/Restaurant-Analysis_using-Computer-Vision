"""3.1 Video im Browser: Das bearbeitete Bild als MJPEG-Stream über FastAPI.
Start: python phase3/3_1_video_in_browser.py  →  http://localhost:8000"""
import sys
import threading
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.utils import VideoSource, draw_box, draw_panel, load_config  # noqa: E402

cfg = load_config()
latest = {"jpg": None}


def worker():
    model = load_model(cfg["model"])
    src = VideoSource(cfg["source"], cfg["process_width"])
    while (frame := src.read()) is not None:
        people = remove_duplicates(detect_people(model, frame, cfg["conf"]))
        for p in people:
            draw_box(frame, p.box)
        draw_panel(frame, [f"Personen: {len(people)}"])
        latest["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()


app = FastAPI()


@app.get("/")
def page():
    return HTMLResponse("<body style='margin:0;background:#111'><img src='/video' style='width:100%'></body>")


@app.get("/video")
def video():
    def gen():
        while True:
            if latest["jpg"]:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest["jpg"] + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
