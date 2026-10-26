"""End-to-end check with the real YOLO models (needs ``ultralytics`` and the model weights).

Builds a short test video from the two sample photos that ship with Ultralytics
(``bus.jpg``: people standing in the street, ``zidane.jpg``: two people), then

1. runs the full :class:`~rcv.analyzer.RestaurantAnalyzer` pipeline on it and
   checks that people, keypoints and postures are found, and
2. runs every OpenCV learning step (1.1 – 2.7) headless for a few frames.

Used by the CI workflow inside the Docker image::

    python tests/e2e_yolo.py                      # full check
    python tests/e2e_yolo.py --make-video out.mp4 # only write the test video
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT, FPS = 1280, 720, 10


def sample_images() -> list[np.ndarray]:
    """The sample photos bundled with Ultralytics, letterboxed to 1280×720."""
    from ultralytics.utils import ASSETS

    images = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(ASSETS / name))
        assert img is not None, f"Beispielbild fehlt: {ASSETS / name}"
        scale = min(WIDTH / img.shape[1], HEIGHT / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        y, x = (HEIGHT - img.shape[0]) // 2, (WIDTH - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        images.append(canvas)
    return images


def make_video(path: Path, seconds_per_image: int = 3) -> Path:
    """Write a short MP4 that shows each sample photo for a few seconds."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for image in sample_images():
        for _ in range(seconds_per_image * FPS):
            writer.write(image)
    writer.release()
    return path


def write_config(video: Path, path: Path) -> None:
    """Test configuration: two tables covering the left and right half of the frame."""
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config.update(source=str(video), occupy_after_sec=0.5, free_after_sec=1.0, lock_window_sec=0.5,
                  tables=[
                      {"id": "L", "seats": 4, "polygon": [[0, 0], [640, 0], [640, 720], [0, 720]]},
                      {"id": "R", "seats": 4, "polygon": [[640, 0], [1280, 0], [1280, 720], [640, 720]]},
                  ])
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def check_pipeline(config_path: Path) -> None:
    """Real YOLO pose model through the whole analysis pipeline."""
    from rcv import Config, RestaurantAnalyzer, VideoSource

    config = Config.load(config_path)
    analyzer = RestaurantAnalyzer(config)
    most_people, with_keypoints, postures = 0, 0, set()
    with VideoSource(config.source, config.process_width, loop=False) as source:
        for frame in source:
            result = analyzer.process(frame, source.timestamp)
            most_people = max(most_people, len(result.people))
            with_keypoints += sum(p.keypoints is not None for p in result.people)
            postures.update(p.posture.value for p in result.people)
            summary = analyzer.summary(result)

    print(f"  Pipeline: max. {most_people} Personen pro Bild, {with_keypoints} mit Keypoints, "
          f"Haltungen {sorted(postures)}, Tische belegt {summary['tables_occupied']}/{summary['tables_total']}")
    assert most_people >= 2, "YOLO hat in den Beispielbildern keine Personen gefunden"
    assert with_keypoints > 0, "Pose-Modell liefert keine Keypoints"
    assert postures - {"unknown"}, "Keine Haltung (sitzt/steht) erkannt"
    json.dumps(summary)  # must be JSON-serialisable for the dashboard


def check_learning_steps(frames_per_step: int = 8) -> None:
    """Run every OpenCV learning step headless: imshow is skipped, 'q' is pressed after N frames."""
    shown = {"n": 0}

    def fake_imshow(*_args):
        shown["n"] += 1

    cv2.imshow = fake_imshow
    cv2.waitKey = lambda *_args: ord("q") if shown["n"] >= frames_per_step else -1
    cv2.destroyAllWindows = cv2.destroyWindow = lambda *_args: None

    scripts = sorted((ROOT / "phase1").glob("*.py")) + sorted((ROOT / "phase2").glob("*.py"))
    for script in scripts:
        shown["n"] = 0
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
        assert shown["n"] == frames_per_step, f"{script.name}: nur {shown['n']} Bilder"
        print(f"  OK  {script.name}")
    (ROOT / "visits.csv").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--make-video", type=Path, help="nur das Testvideo schreiben und beenden")
    args = parser.parse_args()

    if args.make_video:
        print("Testvideo:", make_video(args.make_video))
        return

    with tempfile.TemporaryDirectory() as tmp:
        video = make_video(Path(tmp) / "e2e.mp4")
        config_path = Path(tmp) / "config.json"
        write_config(video, config_path)

        # The learning steps load the default config → point it to the test config.
        os.environ["RCV_CONFIG"] = str(config_path)
        sys.path.insert(0, str(ROOT))

        print("1. Pipeline mit echtem YOLO11-Pose")
        check_pipeline(config_path)
        print("2. Lernschritte 1.1 – 2.7")
        check_learning_steps()
    print("\nEnd-to-End-Test bestanden.")


if __name__ == "__main__":
    main()
