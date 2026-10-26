"""Entry point of the project.

    python main.py            start the dashboard (http://localhost:8000)
    python main.py tables     draw the tables on the video
    python main.py test       run the logic tests (no YOLO needed)
    python main.py 2.7        run a single learning step (1.1 … 3.3)
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class CommandLine:
    """Maps a command name to the script or module that implements it."""

    STEP_FOLDERS = ("phase1", "phase2", "phase3")

    def __init__(self) -> None:
        # "phase2/2_7_table_timer.py" → step "2.7"
        self.steps: dict[str, Path] = {}
        for folder in self.STEP_FOLDERS:
            for script in sorted((ROOT / folder).glob("[0-9]_[0-9]_*.py")):
                major, minor = script.name.split("_")[:2]
                self.steps[f"{major}.{minor}"] = script

    def run(self, argv: list[str]) -> None:
        command = argv[1] if len(argv) > 1 else "dashboard"
        sys.argv = argv[:1] + argv[2:]      # pass remaining arguments to the script

        if command == "dashboard":
            runpy.run_module("app.server", run_name="__main__")
        elif command == "tables":
            runpy.run_path(str(ROOT / "tools" / "mark_tables.py"), run_name="__main__")
        elif command == "test":
            runpy.run_path(str(ROOT / "tests" / "test_logic.py"), run_name="__main__")
        elif command in self.steps:
            runpy.run_path(str(self.steps[command]), run_name="__main__")
        else:
            self.print_help()

    def print_help(self) -> None:
        print(__doc__)
        print("Lernschritte:", ", ".join(self.steps))


if __name__ == "__main__":
    CommandLine().run(sys.argv)
