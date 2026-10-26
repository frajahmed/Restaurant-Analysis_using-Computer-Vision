"""Step 3.3 – Database.

Shows how visits are stored in SQLite and read back. The script fills a demo
database (``demo.db``) with random data and prints some reports. The dashboard
in ``app/`` does exactly this automatically.
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import Database  # noqa: E402
from rcv.tables import Visit  # noqa: E402


class DatabaseDemo:
    """Fills a database with fake visits/snapshots and prints the reports."""

    def __init__(self, path: Path):
        self.db = Database(path)

    def fill(self, visits: int = 20, hours: int = 6) -> None:
        """Insert random visits and one snapshot every 5 minutes."""
        self.db.clear()
        now = time.time()
        for _ in range(visits):
            start = now - random.randint(600, hours * 3600)
            self.db.add_visit(Visit(f"T{random.randint(1, 3)}", start,
                                    start + random.randint(900, 5400), random.randint(1, 4)))
        for minutes in range(0, hours * 60, 5):
            self.db.add_snapshot(now - minutes * 60, random.randint(0, 10), random.randint(0, 3), 3)

    def report(self) -> None:
        print("Heute:", self.db.today_summary())
        print("\nLetzte 5 Besuche:")
        for v in self.db.recent_visits(5):
            started = time.strftime("%H:%M", time.localtime(v["start_ts"]))
            print(f"  {v['table_id']}  {started}  {v['duration'] / 60:5.1f} min  {v['guests']} Gäste")
        print("\nPro Stunde:")
        for h in self.db.hourly():
            if h["avg_guests"] or h["visits"]:
                print(f"  {h['hour']:02d}:00  Ø {h['avg_guests']:4.1f} Gäste  {h['visits']} Besuche")


if __name__ == "__main__":
    demo = DatabaseDemo(ROOT / "demo.db")
    demo.fill()
    demo.report()
