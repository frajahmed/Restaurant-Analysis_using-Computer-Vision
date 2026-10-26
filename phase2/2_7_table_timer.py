"""Step 2.7 – Table timer.

Every visit gets a start, end, duration and guest count. Finished visits are
printed to the console and appended to ``visits.csv``.
"""
import csv
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.analyzer import RestaurantAnalyzer  # noqa: E402
from rcv.config import ROOT  # noqa: E402
from rcv.step import VideoStep  # noqa: E402
from rcv.tables import Visit  # noqa: E402
from rcv.render import format_duration  # noqa: E402


class VisitLog:
    """Writes finished visits to a CSV file."""

    HEADER = ["tisch", "start", "ende", "dauer_sek", "gaeste"]

    def __init__(self, path: Path):
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def write(self, visit: Visit) -> None:
        self._writer.writerow([
            visit.table_id,
            str(timedelta(seconds=int(visit.start))),
            str(timedelta(seconds=int(visit.end))),
            round(visit.duration, 1),
            visit.guests,
        ])
        self._file.flush()      # keep the file usable while the script is running

    def close(self) -> None:
        self._file.close()


class TableTimer(VideoStep):
    title = "2.7 Table Timer"

    def __init__(self):
        super().__init__()
        self.analyzer = RestaurantAnalyzer(self.config)
        self.log = VisitLog(ROOT / "visits.csv")
        self.visits: list[Visit] = []

    def process(self, frame, now):
        result = self.analyzer.process(frame, now)
        tables = self.analyzer.tables

        for visit in result.finished_visits:
            self.visits.append(visit)
            self.log.write(visit)
            print(f"[Besuch] {visit.table_id}: {visit.guests} Gäste, {format_duration(visit.duration)}")

        self.renderer.tables(frame, tables, now)
        self.renderer.people(frame, result.people)

        average = sum(v.duration for v in self.visits) / len(self.visits) if self.visits else 0.0
        self.renderer.panel(frame, [
            f"Gaeste: {tables.guests} / {self.config.total_seats}",
            f"Tische belegt: {tables.occupied_count}/{len(tables)}",
            f"Besuche: {len(self.visits)}  Schnitt: {format_duration(average)}",
        ])
        return frame

    def on_finish(self) -> None:
        """Close the CSV file when the window is closed."""
        self.log.close()


if __name__ == "__main__":
    TableTimer().run()
