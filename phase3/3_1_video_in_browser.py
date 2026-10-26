"""Step 3.1 – Video in the browser.

The processed frames are streamed as MJPEG through FastAPI.
Start: ``python main.py 3.1`` → open http://localhost:8000
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402

from rcv.config import Config  # noqa: E402
from rcv.detection import DuplicateFilter, PersonDetector  # noqa: E402
from rcv.render import Renderer  # noqa: E402
from rcv.streaming import MJPEG_MEDIA_TYPE, BackgroundWorker, FrameBuffer, encode_jpeg  # noqa: E402
from rcv.video import VideoSource  # noqa: E402

PAGE = "<body style='margin:0;background:#111'><img src='/video' style='width:100%'></body>"


class PeopleStream:
    """Detects people in the background and keeps the latest annotated frame."""

    def __init__(self, config: Config):
        self.config = config
        self.frames = FrameBuffer()
        self.worker = BackgroundWorker(self._loop)
        self.renderer = Renderer()

    def _loop(self, stop: threading.Event) -> None:
        detector = PersonDetector(self.config.model, self.config.conf)
        duplicates = DuplicateFilter(self.config.dedupe_iou, self.config.dedupe_contain)
        with VideoSource(self.config.source, self.config.process_width) as source:
            for frame in source:
                if stop.is_set():
                    break
                people = duplicates.apply(detector.detect(frame))
                for person in people:
                    self.renderer.box(frame, person.box)
                self.renderer.panel(frame, [f"Personen: {len(people)}"])
                self.frames.put(encode_jpeg(frame))


def create_app(stream: PeopleStream) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def page():
        return HTMLResponse(PAGE)

    @app.get("/video")
    def video():
        return StreamingResponse(stream.frames.mjpeg_stream(), media_type=MJPEG_MEDIA_TYPE)

    return app


if __name__ == "__main__":
    people_stream = PeopleStream(Config.load())
    people_stream.worker.start()
    uvicorn.run(create_app(people_stream), host="0.0.0.0", port=8000)
