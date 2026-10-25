"""Personenerkennung mit YOLO11, Duplikat-Filter und Sitzen/Stehen per Pose."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import containment, iou

PERSON = 0  # COCO-Klasse "person"

# COCO-Keypoints (YOLO-Pose)
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


@dataclass(eq=False)  # Vergleich per Identität (numpy-Felder sind nicht vergleichbar)
class Person:
    box: tuple            # x1, y1, x2, y2
    conf: float
    keypoints: np.ndarray | None = None   # (17, 3): x, y, Konfidenz
    posture: str = "unknown"              # "sitting" | "standing" | "unknown"
    table: str | None = None
    extra: dict = field(default_factory=dict)


def load_model(name: str):
    from ultralytics import YOLO  # Import hier, damit Logik-Tests ohne Torch laufen
    return YOLO(name)


def detect_people(model, frame, conf=0.4) -> list[Person]:
    """Nur die Klasse 'person' behalten. Funktioniert mit Detect- und Pose-Modell."""
    res = model(frame, classes=[PERSON], conf=conf, verbose=False)[0]
    people = []
    if res.boxes is None:
        return people
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    kps = None
    if getattr(res, "keypoints", None) is not None and res.keypoints.data is not None:
        kps = res.keypoints.data.cpu().numpy()
    for i, (b, c) in enumerate(zip(boxes, confs)):
        people.append(Person(tuple(float(v) for v in b), float(c),
                             kps[i] if kps is not None and i < len(kps) else None))
    return people


# ---------------------------------------------------------------- 2.1 Duplikate
def remove_duplicates(people: list[Person], iou_thr=0.6, contain_thr=0.8) -> list[Person]:
    """Entfernt doppelte Boxen: starke Überlappung (IoU) oder Box steckt fast
    ganz in einer anderen (z.B. Oberkörper-Box in Ganzkörper-Box).
    Die Box mit höherer Konfidenz gewinnt."""
    kept: list[Person] = []
    for p in sorted(people, key=lambda p: p.conf, reverse=True):
        dup = False
        for k in kept:
            if (iou(p.box, k.box) > iou_thr
                    or containment(p.box, k.box) > contain_thr
                    or containment(k.box, p.box) > contain_thr):
                dup = True
                break
        if not dup:
            kept.append(p)
    return kept


# ---------------------------------------------------------------- 2.4 Sitzen/Stehen
def _mid(kp, a, b, min_conf):
    pts = [kp[i][:2] for i in (a, b) if kp[i][2] >= min_conf]
    return np.mean(pts, axis=0) if pts else None


def classify_posture(person: Person, sit_ratio=0.55, min_conf=0.35) -> str:
    """Sitzend, wenn der Oberschenkel eher waagerecht ist:
    vertikaler Abstand Hüfte→Knie ist klein im Verhältnis zur Rumpflänge.
    Sind die Knie verdeckt (typisch unter dem Tisch), entscheidet das
    Seitenverhältnis der Box (Sitzende sind gedrungener)."""
    kp = person.keypoints
    x1, y1, x2, y2 = person.box
    box_ratio = (y2 - y1) / max(1.0, x2 - x1)

    if kp is None:
        return "unknown"
    sho = _mid(kp, L_SHO, R_SHO, min_conf)
    hip = _mid(kp, L_HIP, R_HIP, min_conf)
    knee = _mid(kp, L_KNEE, R_KNEE, min_conf)
    ank = _mid(kp, L_ANK, R_ANK, min_conf)

    if sho is None or hip is None:
        return "standing" if box_ratio > 2.4 else "unknown"
    torso = max(1.0, hip[1] - sho[1])

    if knee is not None:
        thigh_dy = knee[1] - hip[1]
        if thigh_dy / torso < sit_ratio:
            return "sitting"
        if ank is not None and (ank[1] - hip[1]) / torso > 1.4:
            return "standing"
        return "standing" if thigh_dy / torso > 0.85 else "sitting"

    # Knie unsichtbar → Box-Form
    return "standing" if box_ratio > 2.6 else "sitting"
