# Restaurant Analysis using Computer Vision

Ein Restaurant-Analysesystem mit **YOLO11 + Python**: vom rohen Überwachungsvideo bis zum
Live-Dashboard im Browser, das weiß, **welche Tische belegt sind, von wie vielen Gästen und
wie lange**. Kein eigenes Training nötig – es werden die vortrainierten YOLO11-Modelle verwendet.

Inspiriert vom Kurs *„Restaurant Analysis using Computer Vision – Full Course“* (Murtaza's Workshop).
Der Code hier ist eine eigenständige Umsetzung derselben Idee und Kapitelstruktur.

![Stack](https://img.shields.io/badge/Python-3.10+-blue) ![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-purple) ![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-teal)

---

## Funktionen

- Personenerkennung mit YOLO11, Filter nach Klasse, Konfidenz und Überlappung
- Doppelte Boxen entfernen (IoU + „Box in Box“)
- Tische als Polygone im Kamerabild markieren
- Tisch gilt erst nach einigen Sekunden als belegt (Vorbeilaufen zählt nicht) und erst nach
  einigen Sekunden Leere als frei (kurz verdeckt zählt nicht)
- **Kellner ignorieren**: Sitzen/Stehen über Pose-Keypoints (Hüfte, Knie, Knöchel)
- Gäste pro Tisch, **geglättet** (häufigster Wert im Zeitfenster, „locked count“)
- Timer pro Besuch, Protokoll mit Start, Ende, Dauer, Gästezahl
- **Browser-Dashboard** (Handy/Laptop im selben WLAN): Live-Video, Kennzahlen,
  Grundriss mit belegten Plätzen, Stunden-Verlauf, letzte Besuche
- **SQLite**-Datenbank mit Verlauf, **Einstellungsseite** mit Tisch-Editor im Browser

## Projektstruktur

```
├── main.py                   Startpunkt (Dashboard, Tische, Tests, Lernschritte)
├── config.json               Videoquelle, Modelle, Schwellwerte, Tische
├── rcv/                      Kern-Bibliothek (wiederverwendbare Klassen)
│   ├── config.py             Config, TableConfig
│   ├── geometry.py           Box, Polygon
│   ├── video.py              VideoSource
│   ├── detection.py          PersonDetector, DuplicateFilter, PostureClassifier, Person
│   ├── tables.py             Table, TableManager, CountSmoother, Visit, TableState
│   ├── analyzer.py           RestaurantAnalyzer (komplette Pipeline)
│   ├── render.py             Renderer, Window, Color
│   ├── step.py               VideoStep (Basisklasse der Lernschritte)
│   └── streaming.py          FrameBuffer, BackgroundWorker (Video im Browser)
├── phase1/                   1.1 – 1.4  Python-Grundlagen
├── phase2/                   2.1 – 2.7  Tisch-Logik
├── phase3/                   3.1 – 3.3  Browser-Grundlagen
├── app/                      3.4 Komplettes Dashboard
│   ├── server.py             FastAPI-App (create_app)
│   ├── engine.py             Engine – Analyse im Hintergrund-Thread
│   ├── db.py                 Database – SQLite
│   └── static/               index.html, settings.html, style.css
├── tools/mark_tables.py      TableMarker – Tische per Mausklick markieren
└── tests/test_logic.py       Tests für die Logik (ohne YOLO)
```

## Architektur

Der Code ist objektorientiert aufgebaut: Jede Klasse hat genau eine Aufgabe, und alle
Klassen sind mit englischen Docstrings, Kommentaren und Type-Hints dokumentiert.

```
VideoSource ──frame──▶ RestaurantAnalyzer ──────────────────────────────▶ FrameResult
                         │
                         ├─ PersonDetector     YOLO11 → list[Person]
                         ├─ DuplicateFilter    doppelte Boxen entfernen
                         ├─ PostureClassifier  sitzt / steht (Pose-Keypoints)
                         └─ TableManager       Personen → Tische
                              └─ Table         Zustand + CountSmoother + Visit
```

| Klasse | Aufgabe |
|---|---|
| `Config` | lädt, prüft und speichert `config.json` |
| `VideoSource` | Datei, Webcam oder Stream; liefert Bilder und Zeitstempel |
| `PersonDetector` | YOLO-Modell, gibt nur Personen zurück |
| `DuplicateFilter` | entfernt doppelte Boxen (IoU, Box-in-Box) |
| `PostureClassifier` | erkennt Sitzen/Stehen aus Hüfte, Knie, Knöchel |
| `Table` | Zustandsautomat `frei → wartet → belegt → geht? → frei` |
| `CountSmoother` | stabile Gästezahl (häufigster Wert im Zeitfenster) |
| `TableManager` | ordnet Personen den Tischen zu, aktualisiert alle Tische |
| `RestaurantAnalyzer` | verbindet alles zu einer Pipeline |
| `Renderer` / `Window` | zeichnet Ergebnisse, zeigt das Fenster |
| `VideoStep` | Basisklasse: jeder Lernschritt implementiert nur `process()` |
| `Engine` / `Database` | Hintergrund-Analyse und SQLite für das Dashboard |

Der `RestaurantAnalyzer` bekommt den Detector von außen übergeben (Dependency Injection).
Deshalb laufen die Tests mit einem simulierten Detector, ohne YOLO und ohne Grafikkarte.

## Installation

```bash
git clone https://github.com/frajahmed/Restaurant-Analysis_using-Computer-Vision.git
cd Restaurant-Analysis_using-Computer-Vision
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Die YOLO-Gewichte (`yolo11n.pt`, `yolo11n-pose.pt`) lädt Ultralytics beim ersten Start automatisch herunter.
Mit NVIDIA-GPU vorher die passende PyTorch-Version von pytorch.org installieren – deutlich schneller.

## Video vorbereiten

1. Ein Restaurant-/Café-Video aus fester Kameraperspektive (z.B. lizenzfreie Clips von Pexels
   oder Pixabay, Suchbegriff „restaurant cctv“ / „cafe top view“) als `videos/restaurant.mp4` ablegen.
   Alternativ in `config.json` eine Webcam (`"source": "0"`) oder einen RTSP-Stream eintragen.
2. Tische markieren – entweder
   - `python tools/mark_tables.py` (Klick = Punkt, ENTER = fertig, Ziffer = Sitzplätze, `s` = speichern), oder
   - später im Dashboard unter **Einstellungen**.

## Die Schritte

| Schritt | Skript | Neue Idee |
|---|---|---|
| 1.1 | `phase1/1_1_yolo_basics.py` | YOLO11 auf ein Video, Modellgrößen n/s/m/l/x |
| 1.2 | `phase1/1_2_people_detection.py` | nur Klasse „person“, Mindest-Konfidenz |
| 1.3 | `phase1/1_3_people_counting.py` | Personen zählen |
| 1.4 | `phase1/1_4_capacity_readout.py` | Anzahl vs. Sitzplätze, Auslastungsbalken |
| 2.1 | `phase2/2_1_remove_duplicates.py` | doppelte Boxen entfernen |
| 2.2 | `phase2/2_2_table_occupancy.py` | Tische als Polygone, Person → Tisch |
| 2.3 | `phase2/2_3_occupancy_timer.py` | Wartezeit bevor belegt / frei |
| 2.4 | `phase2/2_4_standing_filter.py` | Pose-Modell, Stehende (Kellner) ignorieren |
| 2.5 | `phase2/2_5_people_per_table.py` | Gäste pro Tisch |
| 2.6 | `phase2/2_6_locked_count.py` | stabile Gästezahl |
| 2.7 | `phase2/2_7_table_timer.py` | Besuchs-Timer + `visits.csv` |
| 3.1 | `phase3/3_1_video_in_browser.py` | MJPEG-Stream mit FastAPI |
| 3.2 | `phase3/3_2_live_numbers.py` | Live-Zahlen als JSON |
| 3.3 | `phase3/3_3_database.py` | SQLite speichern & auswerten |
| 3.4 | `python main.py` | **komplettes Dashboard** |

Alle Skripte aus dem Projektordner starten, z.B. `python phase2/2_4_standing_filter.py`. Fenster mit `q` schließen.

## Dashboard starten

```bash
python main.py
```

Weitere Befehle: `python main.py tables` (Tische markieren), `python main.py test` (Tests), `python main.py 2.7` (einzelner Lernschritt).

- Am Rechner: <http://localhost:8000>
- Vom Handy im selben WLAN: die beim Start angezeigte Adresse `http://<IP>:8000`
  (ggf. Port 8000 in der Firewall freigeben)

**API**: `/api/stats` · `/api/visits` · `/api/history` · `/api/config` (GET/POST) · `/api/snapshot` · `/video`

## Wichtige Einstellungen (`config.json`)

| Schlüssel | Bedeutung |
|---|---|
| `conf` | Mindest-Konfidenz der Erkennung |
| `dedupe_iou`, `dedupe_contain` | Schwellen für doppelte Boxen |
| `occupy_after_sec` | so lange muss jemand sitzen, bis der Tisch „belegt“ ist |
| `free_after_sec` | so lange muss er leer sein, bis er „frei“ ist |
| `lock_window_sec` | Zeitfenster für die geglättete Gästezahl |
| `sit_ratio` | Oberschenkel-/Rumpf-Verhältnis, unter dem jemand als sitzend gilt |
| `process_width` | Verarbeitungsbreite – kleiner = schneller |

## Wie es funktioniert

- **Sitzen vs. Stehen**: Das Pose-Modell liefert 17 Körperpunkte. Ist der vertikale Abstand
  Hüfte→Knie klein im Vergleich zur Rumpflänge (Schulter→Hüfte), liegt der Oberschenkel waagerecht → sitzt.
  Sind die Knie unter dem Tisch verdeckt, entscheidet das Seitenverhältnis der Box.
- **Zuordnung**: Referenzpunkt ist die Hüftmitte (oder 70 % der Boxhöhe). Liegt er im Tisch-Polygon, gehört die Person zum Tisch.
- **Status-Automat** pro Tisch: `frei → wartet → belegt → geht? → frei`.
- **Glättung**: Häufigster Wert der letzten Sekunden; die angezeigte Zahl wechselt erst, wenn ein neuer Wert ein ganzes Fenster lang stabil ist.

## Tests

```bash
python main.py test
```

10 Tests prüfen Geometrie, Duplikat-Filter, Sitzen/Stehen, Tisch-Zuordnung, Glättung,
den kompletten Tisch-Ablauf, die Pipeline und die Konfiguration – ohne YOLO und ohne Video.
