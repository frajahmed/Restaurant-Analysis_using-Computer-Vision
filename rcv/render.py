"""Gemeinsames Zeichnen von Tischen und Personen (Phase 2 + Dashboard)."""
from .tables import FREE, LEAVING, OCCUPIED, PENDING, person_point
from .utils import GREEN, GREY, RED, YELLOW, draw_box, draw_polygon, fmt_duration, put_label

STATE_COLOR = {FREE: GREEN, PENDING: YELLOW, OCCUPIED: RED, LEAVING: (0, 140, 255)}
STATE_TEXT = {FREE: "frei", PENDING: "wartet", OCCUPIED: "belegt", LEAVING: "geht?"}


def draw_people(img, people):
    import cv2
    for p in people:
        if p.posture == "standing":
            draw_box(img, p.box, GREY, "steht")
        else:
            draw_box(img, p.box, YELLOW if p.table else GREEN, p.table)
        x, y = person_point(p)
        cv2.circle(img, (int(x), int(y)), 4, (255, 0, 255), -1)


def draw_tables(img, tables, now, show_guests=True, show_timer=True):
    for t in tables:
        color = STATE_COLOR[t.state]
        draw_polygon(img, t.polygon, color, 0.22)
        parts = [t.id, STATE_TEXT[t.state]]
        if show_guests and t.occupied:
            parts.append(f"{t.guests}/{t.seats} P")
        if show_timer and t.occupied:
            parts.append(fmt_duration(t.duration(now)))
        top = min(t.polygon, key=lambda p: p[1])
        put_label(img, " | ".join(parts), (top[0], top[1] - 8), color)
