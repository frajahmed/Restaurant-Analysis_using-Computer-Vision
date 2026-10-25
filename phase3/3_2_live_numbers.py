"""3.2 Live-Zahlen: Zusätzlich zum Video liefert /stats die Werte als JSON,
die Seite fragt sie jede Sekunde ab. Start: python phase3/3_2_live_numbers.py"""
import sys
import threading
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import classify_posture, detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import draw_people, draw_tables  # noqa: E402
from rcv.tables import assign_tables, build_tables  # noqa: E402
from rcv.utils import VideoSource, load_config  # noqa: E402

cfg = load_config()
state = {"jpg": None, "stats": {}}

PAGE = """<body style="margin:0;background:#111;color:#eee;font-family:system-ui">
<div style="display:flex;gap:30px;padding:14px;font-size:22px">
  <div>Gäste: <b id=g>–</b></div><div>Tische belegt: <b id=t>–</b></div></div>
<img src="/video" style="width:100%">
<script>setInterval(async()=>{const s=await (await fetch('/stats')).json();
 g.textContent=s.guests+' / '+s.seats; t.textContent=s.occupied+' / '+s.tables},1000)</script></body>"""


def worker():
    model = load_model(cfg["pose_model"])
    src = VideoSource(cfg["source"], cfg["process_width"])
    tables = build_tables(cfg)
    while (frame := src.read()) is not None:
        now = src.video_time()
        people = remove_duplicates(detect_people(model, frame, cfg["conf"]))
        for p in people:
            p.posture = classify_posture(p, cfg["sit_ratio"])
        per = assign_tables(people, cfg["tables"], only_sitting=True)
        for t in tables:
            t.update(now, len(per[t.id]))
        draw_tables(frame, tables, now)
        draw_people(frame, people)
        state["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()
        state["stats"] = {"guests": sum(t.guests for t in tables), "seats": cfg["seats_total"],
                          "occupied": sum(t.occupied for t in tables), "tables": len(tables)}


app = FastAPI()


@app.get("/")
def page():
    return HTMLResponse(PAGE)


@app.get("/stats")
def stats():
    return state["stats"]


@app.get("/video")
def video():
    def gen():
        while True:
            if state["jpg"]:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["jpg"] + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
