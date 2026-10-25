"""Tische im Kamerabild markieren und in config.json speichern.

Bedienung:
  Linksklick      Punkt setzen (Tisch inkl. Stühle umranden)
  ENTER / n       Polygon abschließen → danach Ziffer 1-9 = Anzahl Sitzplätze
  u               letzten Punkt / letzten Tisch entfernen
  c               alle Tische löschen
  s               speichern
  q / ESC         beenden
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.utils import VideoSource, draw_polygon, load_config, put_label, save_config, GREEN, YELLOW  # noqa: E402

cfg = load_config()
src = VideoSource(cfg["source"], cfg.get("process_width", 1280), loop=False)
frame = None
for _ in range(int(src.fps)):  # 1 s vorspulen, erstes Bild ist oft schwarz
    f = src.read()
    if f is None:
        break
    frame = f
src.release()
if frame is None:
    sys.exit("Konnte kein Bild aus der Videoquelle lesen.")

tables = [dict(t) for t in cfg.get("tables", [])] if "--keep" in sys.argv else []
current: list = []
waiting_seats = False


def on_mouse(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and not waiting_seats:
        current.append([x, y])


cv2.namedWindow("Tische markieren")
cv2.setMouseCallback("Tische markieren", on_mouse)

while True:
    img = frame.copy()
    for t in tables:
        draw_polygon(img, t["polygon"], GREEN)
        put_label(img, f'{t["id"]} ({t["seats"]} Plaetze)', t["polygon"][0], GREEN)
    for p in current:
        cv2.circle(img, tuple(p), 4, YELLOW, -1)
    if len(current) > 1:
        cv2.polylines(img, [__import__("numpy").array(current)], False, YELLOW, 2)
    msg = "Sitzplaetze? Ziffer 1-9" if waiting_seats else "Klick=Punkt  ENTER=fertig  u=undo  s=speichern  q=ende"
    put_label(img, msg, (10, 25), YELLOW, 0.6)
    cv2.imshow("Tische markieren", img)
    k = cv2.waitKey(20) & 0xFF

    if waiting_seats and ord("1") <= k <= ord("9"):
        tables.append({"id": f"T{len(tables) + 1}", "seats": k - ord("0"), "polygon": current.copy()})
        current.clear()
        waiting_seats = False
    elif k in (13, ord("n")) and len(current) >= 3:
        waiting_seats = True
    elif k == ord("u"):
        if current:
            current.pop()
        elif tables:
            tables.pop()
    elif k == ord("c"):
        tables.clear(); current.clear()
    elif k == ord("s"):
        cfg["tables"] = tables
        cfg["seats_total"] = sum(t["seats"] for t in tables) or cfg.get("seats_total", 0)
        save_config(cfg)
        print(f"Gespeichert: {len(tables)} Tische, {cfg['seats_total']} Plätze → config.json")
    elif k in (ord("q"), 27):
        break

cv2.destroyAllWindows()
