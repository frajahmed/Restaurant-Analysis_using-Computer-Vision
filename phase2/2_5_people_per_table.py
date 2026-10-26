"""Step 2.5 – People per table.

Number of seated people inside each table polygon. The raw number still jumps
whenever someone is hidden for a moment – step 2.6 fixes that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.analyzer import RestaurantAnalyzer  # noqa: E402
from rcv.step import VideoStep  # noqa: E402
from rcv.render import Renderer  # noqa: E402


class PeoplePerTable(VideoStep):
    title = "2.5 People Per Table"

    def __init__(self):
        super().__init__()
        # From here on the full pipeline (detect → dedupe → posture → tables) is reused.
        self.analyzer = RestaurantAnalyzer(self.config)

    def process(self, frame, now):
        result = self.analyzer.process(frame, now)
        tables = self.analyzer.tables

        for table in tables:
            color = Renderer.STATE_COLORS[table.state]
            self.renderer.polygon(frame, table.area, color)
            self.renderer.label(frame, f"{table.id}: {table.raw_count}/{table.seats} Gaeste",
                                table.area.top_point, color)

        self.renderer.people(frame, result.people)
        self.renderer.panel(frame, [f"Gaeste an Tischen: {tables.raw_guests}"])
        return frame


if __name__ == "__main__":
    PeoplePerTable().run()
