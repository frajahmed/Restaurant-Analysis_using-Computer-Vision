"""Startpunkt des Projekts.

  python main.py            Dashboard starten (http://localhost:8000)
  python main.py tables     Tische im Video markieren
  python main.py test       Logik-Tests ausführen (ohne YOLO)
  python main.py 2.7        einen Lernschritt starten (1.1 … 3.3)
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STEPS = {p.name.split("_")[0] + "." + p.name.split("_")[1]: p
         for folder in ("phase1", "phase2", "phase3") for p in sorted((ROOT / folder).glob("*.py"))}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    sys.argv = sys.argv[:1] + sys.argv[2:]

    if cmd == "dashboard":
        runpy.run_module("app.server", run_name="__main__")
    elif cmd == "tables":
        runpy.run_path(str(ROOT / "tools" / "mark_tables.py"), run_name="__main__")
    elif cmd == "test":
        runpy.run_path(str(ROOT / "tests" / "test_logic.py"), run_name="__main__")
    elif cmd in STEPS:
        runpy.run_path(str(STEPS[cmd]), run_name="__main__")
    else:
        print(__doc__)
        print("Lernschritte:", ", ".join(STEPS))


if __name__ == "__main__":
    main()
