"""3.3 Datenbank: So werden Besuche in SQLite gespeichert und wieder gelesen.
Das Skript legt eine Demo-Datenbank an (demo.db) und gibt Auswertungen aus.
Im Dashboard (app/) passiert genau das automatisch."""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.db import Database  # noqa: E402

db = Database(ROOT / "demo.db")
db.clear()
now = time.time()
for i in range(20):
    start = now - random.randint(600, 6 * 3600)
    db.add_visit(f"T{random.randint(1, 3)}", start, start + random.randint(900, 5400), random.randint(1, 4))
for m in range(0, 360, 5):
    db.add_snapshot(now - m * 60, random.randint(0, 10), random.randint(0, 3), 3)

print("Heute:", db.today_summary())
print("\nLetzte 5 Besuche:")
for v in db.recent_visits(5):
    print(f"  {v['table_id']}  {time.strftime('%H:%M', time.localtime(v['start_ts']))}  "
          f"{v['duration'] / 60:5.1f} min  {v['guests']} Gäste")
print("\nPro Stunde:")
for h in db.hourly():
    if h["avg_guests"] or h["visits"]:
        print(f"  {h['hour']:02d}:00  Ø {h['avg_guests']:4.1f} Gäste  {h['visits']} Besuche")
