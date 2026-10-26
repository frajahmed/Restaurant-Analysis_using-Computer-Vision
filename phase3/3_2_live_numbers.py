"""Step 3.2 – Live numbers.

In addition to the video, ``/stats`` returns the key figures as JSON; the page
polls it once per second. Start: ``python main.py 3.2``
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402

from rcv.analyzer import RestaurantAnalyzer  # noqa: E402
from rcv.config import Config  # noqa: E402
from rcv.render import Renderer  # noqa: E402
from rcv.streaming import MJPEG_MEDIA_TYPE, BackgroundWorker, FrameBuffer, encode_jpeg  # noqa: E402
from rcv.video import VideoSource  # noqa: E402

PAGE = """<body style="margin:0;background:#111;color:#eee;font-family:system-ui">
<div style="display:flex;gap:30px;padding:14px;font-size:22px">
  <div>Gäste: <b id=g>–</b></div><div>Tische belegt: <b id=t>–</b></div></div>
<img src="/video" style="width:100%">
<script>
setInterval(async () => {
  const s = await (await fetch('/stats')).json();
  g.textContent = s.guests + ' / ' + s.seats_total;
  t.textContent = s.tables_occupied + ' / ' + s.tables_total;
}, 1000);
</script></body>"""


class LiveAnalysis:
    """Runs the full pipeline in the background; exposes the frame and the numbers."""

    def __init__(self, config: Config):
        self.config = config
        self.frames = FrameBuffer()
        self.worker = BackgroundWorker(self._loop)
        self.renderer = Renderer()
        self._stats: dict = {}
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _loop(self, stop: threading.Event) -> None:
        analyzer = RestaurantAnalyzer(self.config)
        with VideoSource(self.config.source, self.config.process_width) as source:
            for frame in source:
                if stop.is_set():
                    break
                now = source.timestamp
                result = analyzer.process(frame, now)
                self.renderer.tables(frame, analyzer.tables, now)
                self.renderer.people(frame, result.people)
                self.frames.put(encode_jpeg(frame))
                with self._lock:
                    self._stats = analyzer.summary(result)


def create_app(live: LiveAnalysis) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def page():
        return HTMLResponse(PAGE)

    @app.get("/stats")
    def stats():
        return live.stats

    @app.get("/video")
    def video():
        return StreamingResponse(live.frames.mjpeg_stream(), media_type=MJPEG_MEDIA_TYPE)

    return app


if __name__ == "__main__":
    live_analysis = LiveAnalysis(Config.load())
    live_analysis.worker.start()
    uvicorn.run(create_app(live_analysis), host="0.0.0.0", port=8000)
