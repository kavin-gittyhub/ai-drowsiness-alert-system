#!/usr/bin/env python3
"""
AI Live Drowsiness Alert System
===============================
Real-time drowsiness detection from a webcam feed.

Pipeline
--------
Webcam frame
  -> MediaPipe Face Landmarker (468 facial landmarks, AI model)
  -> Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
  -> Temporal rules: closed-eye timer, PERCLOS, yawn counter, face-lost timer
  -> Loud alarm + on-screen alert + CSV event log + alert snapshots

Controls (click the video window first)
---------------------------------------
  q / Esc   quit
  c         recalibrate (look at the camera with eyes normally open)
  m         mute / unmute the alarm
"""

import argparse
import csv
import os
import sys
import threading
import time
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("MediaPipe is not installed. Run:  pip install -r requirements.txt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
ALARM_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")

# MediaPipe Face Mesh landmark indices, ordered p1..p6 for the EAR formula:
#   p1 / p4 = eye corners, (p2, p6) and (p3, p5) = upper/lower eyelid pairs
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Inner-lip points: corners (78, 308) and three vertical pairs
MOUTH_CORNERS = (78, 308)
MOUTH_PAIRS = [(81, 178), (13, 14), (311, 402)]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    camera: int = 0
    ear_threshold: Optional[float] = None  # None = auto-calibrate per user
    closed_seconds: float = 1.0            # eyes closed this long -> ALARM
    mar_threshold: float = 0.6             # mouth opening that counts as a yawn
    yawn_seconds: float = 1.0              # mouth must stay open this long
    perclos_window: float = 60.0           # rolling window for PERCLOS (s)
    perclos_limit: float = 0.15            # >15 % eyes-closed time = fatigue
    yawn_limit: int = 3                    # this many yawns ...
    yawn_window: float = 300.0             # ... within 5 minutes = fatigue
    noface_seconds: float = 3.0            # face missing this long -> warning
    alarm_on_no_face: bool = False
    calib_seconds: float = 3.0
    sound: bool = True
    enhance: bool = False                  # CLAHE low-light enhancement
    snapshots: bool = True
    log_dir: str = os.path.join(BASE_DIR, "logs")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="AI Live Drowsiness Alert System")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--ear-threshold", type=float, default=None,
                   help="fixed EAR threshold; skips auto-calibration (e.g. 0.21)")
    p.add_argument("--closed-seconds", type=float, default=1.0,
                   help="seconds of closed eyes before the alarm (default 1.0)")
    p.add_argument("--mar-threshold", type=float, default=0.6,
                   help="mouth aspect ratio counted as a yawn (default 0.6)")
    p.add_argument("--alarm-on-no-face", action="store_true",
                   help="also sound the alarm when the face is not visible")
    p.add_argument("--enhance", action="store_true",
                   help="improve low-light / night images with CLAHE")
    p.add_argument("--no-sound", action="store_true", help="disable the alarm sound")
    p.add_argument("--no-snapshots", action="store_true",
                   help="do not save a picture when an alert fires")
    a = p.parse_args()
    return Config(camera=a.camera, ear_threshold=a.ear_threshold,
                  closed_seconds=a.closed_seconds, mar_threshold=a.mar_threshold,
                  alarm_on_no_face=a.alarm_on_no_face, enhance=a.enhance,
                  sound=not a.no_sound, snapshots=not a.no_snapshots)


# --------------------------------------------------------------------------
# Geometry: Eye Aspect Ratio and Mouth Aspect Ratio
# --------------------------------------------------------------------------
def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)  (Soukupova & Cech, 2016).
    Roughly 0.25-0.35 when open, drops towards 0.1 when closed."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Average inner-lip opening divided by mouth width.
    ~0.0-0.1 closed, 0.2-0.4 talking, >0.6 yawning."""
    width = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    vertical = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_PAIRS) / len(MOUTH_PAIRS)
    return vertical / width


# --------------------------------------------------------------------------
# Face landmark detector (MediaPipe)
# --------------------------------------------------------------------------
def download_model() -> bool:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading face landmark model (~3.7 MB, first run only)...")
    try:
        tmp = MODEL_PATH + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
        print("Model saved to", MODEL_PATH)
        return True
    except Exception as exc:  # network blocked, offline, etc.
        print("Could not download the model:", exc)
        return False


class FaceLandmarkDetector:
    """Wraps MediaPipe. Uses the modern Tasks API (mediapipe >= 0.10) and falls
    back to the legacy Face Mesh solution on older installs."""

    def __init__(self):
        self.backend = None
        self._last_ts = -1
        self._t0 = time.monotonic()
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if os.path.exists(MODEL_PATH) or download_model():
                options = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.backend = "tasks"
        except Exception as exc:
            print("MediaPipe Tasks API unavailable:", exc)

        if self.backend is None:
            solutions = getattr(mp, "solutions", None)
            if solutions is not None and hasattr(solutions, "face_mesh"):
                self.mesh = solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.backend = "solutions"

        if self.backend is None:
            sys.exit(
                "No face model available.\nDownload this file manually:\n  "
                f"{MODEL_URL}\nand save it as:\n  {MODEL_PATH}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns an (N, 2) array of landmark pixel coordinates, or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "tasks":
            ts = int((time.monotonic() - self._t0) * 1000)
            ts = max(ts, self._last_ts + 1)  # timestamps must strictly increase
            self._last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(image, ts)
            if not result.face_landmarks:
                return None
            landmarks = result.face_landmarks[0]
        else:
            rgb.flags.writeable = False
            result = self.mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            landmarks = result.multi_face_landmarks[0].landmark
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

    def close(self):
        try:
            (self.landmarker if self.backend == "tasks" else self.mesh).close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Alarm sound
# --------------------------------------------------------------------------
def create_alarm_wav(path: str, sr: int = 44100):
    """Generates a two-tone siren so the project needs no external audio file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def beep(freq, dur):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.01)  # 10 ms fade in/out
        return np.sin(2 * np.pi * freq * t) * env

    gap = np.zeros(int(sr * 0.04))
    tone = np.concatenate([beep(988, 0.22), gap, beep(1318, 0.22), gap])
    audio = (tone * 0.85 * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


class Alarm:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.muted = False
        self.playing = False
        self.backend = "none"
        if not enabled:
            return
        if not os.path.exists(ALARM_PATH):
            create_alarm_wav(ALARM_PATH)
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.init()
            self.sound = pygame.mixer.Sound(ALARM_PATH)
            self.backend = "pygame"
        except Exception:
            if sys.platform.startswith("win"):
                import winsound
                self.winsound = winsound
                self.backend = "winsound"
            else:
                self.backend = "bell"  # terminal bell as a last resort

    def start(self):
        if not self.enabled or self.muted or self.playing:
            return
        self.playing = True
        if self.backend == "pygame":
            self.sound.play(loops=-1)
        elif self.backend == "winsound":
            ws = self.winsound
            ws.PlaySound(ALARM_PATH, ws.SND_FILENAME | ws.SND_ASYNC | ws.SND_LOOP)
        elif self.backend == "bell":
            threading.Thread(target=self._bell_loop, daemon=True).start()

    def _bell_loop(self):
        while self.playing:
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.5)

    def stop(self):
        if not self.playing:
            return
        self.playing = False
        if self.backend == "pygame":
            self.sound.stop()
        elif self.backend == "winsound":
            self.winsound.PlaySound(None, self.winsound.SND_PURGE)

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop()
        return self.muted


# --------------------------------------------------------------------------
# Event logger (CSV)
# --------------------------------------------------------------------------
class EventLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"session_{self.stamp}.csv")
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestamp", "event", "ear", "mar", "perclos", "details"])

    def event(self, name, ear=None, mar=None, perclos=None, details=""):
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        ts = datetime.now().isoformat(timespec="seconds")
        self._w.writerow([ts, name, fmt(ear), fmt(mar), fmt(perclos), details])
        self._f.flush()
        print(f"[{ts}] {name:<16} {details}")

    def close(self):
        self._f.close()


# --------------------------------------------------------------------------
# Drowsiness decision logic
# --------------------------------------------------------------------------
@dataclass
class FrameState:
    status: str
    ear: Optional[float]
    mar: Optional[float]
    perclos: float
    closed_for: float
    calib_left: float = 0.0


class DrowsinessMonitor:
    def __init__(self, cfg: Config, logger: EventLogger):
        self.cfg = cfg
        self.log = logger
        self.ear_threshold = cfg.ear_threshold
        self.ear_hist = deque(maxlen=3)     # light smoothing against jitter
        self.eyes_closed = False
        self.closed_since = None
        self.mouth_open_since = None
        self.face_lost_since = None
        self.perclos_samples = deque()      # (time, eyes_closed)
        self.perclos_since = None
        self.yawn_times = deque()
        self.blinks = 0
        self.yawns = 0
        self.alerts = 0
        self.alert_active = False
        self.alert_started = 0.0
        self.fatigue_active = False
        self.noface_active = False
        self.snapshot_pending = False
        self.active_since = None            # for blink-rate calculation
        self.last_ear = self.last_mar = None
        self.last_perclos = 0.0
        self.calibrating = False
        if self.ear_threshold is None:
            self.start_calibration(time.monotonic())
        else:
            self._begin_monitoring(time.monotonic())

    # -- calibration ------------------------------------------------------
    def start_calibration(self, now):
        self.calibrating = True
        self.calib_start = now
        self.calib_values = []
        self.alert_active = False
        self.closed_since = None
        print("Calibrating: look at the camera with your eyes normally open...")

    def _begin_monitoring(self, now):
        self.perclos_samples.clear()
        self.perclos_since = now
        if self.active_since is None:
            self.active_since = now

    def _finish_calibration(self, now):
        baseline = float(np.median(self.calib_values))
        self.ear_threshold = float(np.clip(baseline * 0.72, 0.12, 0.30))
        self.calibrating = False
        self._begin_monitoring(now)
        self.log.event("CALIBRATED", ear=baseline,
                       details=f"open-eye EAR {baseline:.3f} -> threshold "
                               f"{self.ear_threshold:.3f}")

    # -- main update --------------------------------------------------------
    def update(self, pts: Optional[np.ndarray], now: float) -> FrameState:
        cfg = self.cfg

        # Face not visible ------------------------------------------------
        if pts is None:
            if self.face_lost_since is None:
                self.face_lost_since = now
            if (not self.noface_active
                    and now - self.face_lost_since >= cfg.noface_seconds):
                self.noface_active = True
                self.log.event("NO_FACE", details="face not visible / looking away")
            self.mouth_open_since = None
            self.ear_hist.clear()
            # If the alarm was already ringing (e.g. head dropped forward)
            # it keeps ringing until open eyes are seen again.
            return self._state(now, None, None)

        if self.noface_active:
            self.log.event("FACE_BACK")
        self.face_lost_since = None
        self.noface_active = False

        ear_raw = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2
        self.ear_hist.append(ear_raw)
        ear = float(np.mean(self.ear_hist))
        mar = mouth_aspect_ratio(pts)
        self.last_ear, self.last_mar = ear, mar

        if self.calibrating:
            self.calib_values.append(ear)
            if (now - self.calib_start >= cfg.calib_seconds
                    and len(self.calib_values) >= 15):
                self._finish_calibration(now)
            return self._state(now, ear, mar)

        # Eye state with hysteresis (prevents flicker near the threshold) --
        thr = self.ear_threshold
        if self.eyes_closed:
            self.eyes_closed = ear < thr * 1.05
        else:
            self.eyes_closed = ear < thr

        # Yawn detection ----------------------------------------------------
        if mar > cfg.mar_threshold:
            if self.mouth_open_since is None:
                self.mouth_open_since = now
        elif self.mouth_open_since is not None:
            if now - self.mouth_open_since >= cfg.yawn_seconds:
                self.yawns += 1
                self.yawn_times.append(now)
                self.log.event("YAWN", ear, mar, self.last_perclos,
                               f"yawn #{self.yawns}")
            self.mouth_open_since = None
        yawning = self.mouth_open_since is not None

        # Closed-eye timer -> drowsiness alarm --------------------------------
        if self.eyes_closed:
            if self.closed_since is None:
                self.closed_since = now
            closed_for = now - self.closed_since
            # Eyes squint during a big yawn, so do not fire mid-yawn.
            if closed_for >= cfg.closed_seconds and not self.alert_active and not yawning:
                self.alert_active = True
                self.alert_started = now
                self.alerts += 1
                self.snapshot_pending = cfg.snapshots
                self.log.event("DROWSY_ALERT", ear, mar, self.last_perclos,
                               f"eyes closed {closed_for:.1f}s (alert #{self.alerts})")
        else:
            if self.closed_since is not None and now - self.closed_since < cfg.closed_seconds:
                self.blinks += 1
            self.closed_since = None
            if self.alert_active:
                self.alert_active = False
                self.log.event("ALERT_CLEARED", ear, mar, self.last_perclos,
                               f"alarm lasted {now - self.alert_started:.1f}s")

        # PERCLOS: % of time eyes were closed in the rolling window ----------
        self.perclos_samples.append((now, self.eyes_closed))
        while self.perclos_samples and now - self.perclos_samples[0][0] > cfg.perclos_window:
            self.perclos_samples.popleft()
        closed_n = sum(1 for _, c in self.perclos_samples if c)
        self.last_perclos = closed_n / max(1, len(self.perclos_samples))

        # Fatigue = high PERCLOS (after 20 s of data) or frequent yawning ----
        while self.yawn_times and now - self.yawn_times[0] > cfg.yawn_window:
            self.yawn_times.popleft()
        enough_data = now - self.perclos_since >= 20.0
        fatigue = ((enough_data and self.last_perclos >= cfg.perclos_limit)
                   or len(self.yawn_times) >= cfg.yawn_limit)
        if fatigue and not self.fatigue_active:
            reason = (f"PERCLOS {self.last_perclos:.0%}"
                      if self.last_perclos >= cfg.perclos_limit
                      else f"{len(self.yawn_times)} yawns in {cfg.yawn_window / 60:.0f} min")
            self.log.event("FATIGUE_WARNING", ear, mar, self.last_perclos, reason)
        self.fatigue_active = fatigue

        return self._state(now, ear, mar)

    def _state(self, now, ear, mar) -> FrameState:
        closed_for = (now - self.closed_since) if self.closed_since else 0.0
        if self.alert_active:
            status = "DROWSY"
        elif self.noface_active:
            status = "NO FACE"
        elif self.calibrating:
            status = "CALIBRATING"
        elif self.fatigue_active:
            status = "FATIGUE"
        elif (self.mouth_open_since is not None
              and now - self.mouth_open_since >= self.cfg.yawn_seconds):
            status = "YAWNING"
        else:
            status = "AWAKE"
        calib_left = (max(0.0, self.cfg.calib_seconds - (now - self.calib_start))
                      if self.calibrating else 0.0)
        return FrameState(status, ear, mar, self.last_perclos, closed_for, calib_left)

    def blink_rate(self, now) -> float:
        if self.active_since is None:
            return 0.0
        minutes = (now - self.active_since) / 60
        return self.blinks / minutes if minutes > 0.1 else 0.0


# --------------------------------------------------------------------------
# Drawing / HUD
# --------------------------------------------------------------------------
STATUS_STYLE = {  # BGR colour, banner text
    "AWAKE":       ((70, 190, 70),  "Status: AWAKE"),
    "CALIBRATING": ((200, 160, 0),  "Calibrating - keep eyes open, look at camera"),
    "YAWNING":     ((0, 210, 255),  "Yawn detected"),
    "FATIGUE":     ((0, 140, 255),  "Fatigue signs - please take a break"),
    "NO FACE":     ((0, 200, 255),  "Face not detected - look ahead"),
    "DROWSY":      ((0, 0, 255),    "DROWSINESS ALERT!  WAKE UP!"),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def draw_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, pts, state: FrameState, mon: DrowsinessMonitor,
                 fps: float, muted: bool, now: float):
    h, w = frame.shape[:2]
    color, message = STATUS_STYLE[state.status]

    # Eye and mouth outlines
    if pts is not None:
        eye_col = (0, 0, 255) if mon.eyes_closed else (0, 255, 0)
        for idx in (LEFT_EYE, RIGHT_EYE):
            poly = pts[idx].astype(np.int32)
            cv2.polylines(frame, [poly], True, eye_col, 1, cv2.LINE_AA)
        mouth = pts[[78, 81, 13, 311, 308, 402, 14, 178]].astype(np.int32)
        mouth_col = (0, 210, 255) if mon.mouth_open_since else (255, 200, 0)
        cv2.polylines(frame, [mouth], True, mouth_col, 1, cv2.LINE_AA)

    # Metrics panel
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (262, 200), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
    thr = mon.ear_threshold
    lines = [
        f"EAR: {state.ear:.3f}" if state.ear is not None else "EAR: --",
        f"EAR threshold: {thr:.3f}" if thr else "EAR threshold: calibrating",
        f"MAR: {state.mar:.2f}" if state.mar is not None else "MAR: --",
        f"PERCLOS (60s): {state.perclos:.0%}",
        f"Blinks: {mon.blinks}  ({mon.blink_rate(now):.0f}/min)",
        f"Yawns: {mon.yawns}    Alerts: {mon.alerts}",
        f"FPS: {fps:.0f}" + ("    [MUTED]" if muted else ""),
    ]
    for i, line in enumerate(lines):
        draw_text(frame, line, (18, 34 + i * 24))

    # Closed-eye progress bar (fills up to the alarm point)
    if state.closed_for > 0 and not mon.calibrating:
        frac = min(1.0, state.closed_for / mon.cfg.closed_seconds)
        x0, y0, bw = w - 230, 20, 210
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + 16), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bw * frac), y0 + 16), (0, 0, 255), -1)
        draw_text(frame, f"Eyes closed {state.closed_for:.1f}s", (x0, y0 + 36), 0.5)

    # Status banner
    if state.status == "CALIBRATING":
        message += f" ({state.calib_left:.0f}s)"
    flash = state.status == "DROWSY" and int(now * 4) % 2 == 0
    banner_col = (255, 255, 255) if flash else color
    text_col = (0, 0, 255) if flash else (255, 255, 255)
    cv2.rectangle(frame, (0, h - 50), (w, h), banner_col, -1)
    scale = 0.9 if state.status == "DROWSY" else 0.65
    (tw, _), _ = cv2.getTextSize(message, FONT, scale, 2)
    cv2.putText(frame, message, ((w - tw) // 2, h - 17), FONT, scale, text_col, 2, cv2.LINE_AA)
    if state.status == "DROWSY":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 51), (0, 0, 255), 8)

    draw_text(frame, "q: quit   c: recalibrate   m: mute", (w - 300, h - 60), 0.45, (220, 220, 220))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def open_camera(index: int):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # faster start on Windows
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --camera 1, or close other "
                 "apps that use the webcam.")
    return cap


def main():
    cfg = parse_args()
    print("=" * 56)
    print(" AI Live Drowsiness Alert System")
    print("=" * 56)

    detector = FaceLandmarkDetector()
    print("Face model backend:", detector.backend)
    alarm = Alarm(cfg.sound)
    print("Alarm sound:", alarm.backend if cfg.sound else "disabled")
    logger = EventLogger(cfg.log_dir)
    snap_dir = os.path.join(cfg.log_dir, f"snapshots_{logger.stamp}")
    logger.event("SESSION_START", details=f"camera {cfg.camera}")

    cap = open_camera(cfg.camera)
    monitor = DrowsinessMonitor(cfg, logger)
    window = "AI Drowsiness Alert System"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)

    fps, last_t, failures = 0.0, time.monotonic(), 0
    session_start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 30:
                    print("Camera stopped sending frames.")
                    break
                continue
            failures = 0

            frame = cv2.flip(frame, 1)  # mirror view feels natural
            if cfg.enhance:
                frame = enhance_lighting(frame)

            pts = detector.detect(frame)
            now = time.monotonic()
            state = monitor.update(pts, now)

            ring = state.status == "DROWSY" or (
                cfg.alarm_on_no_face and state.status == "NO FACE")
            alarm.start() if ring else alarm.stop()

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_overlay(frame, pts, state, monitor, fps, alarm.muted, now)

            if monitor.snapshot_pending:
                os.makedirs(snap_dir, exist_ok=True)
                name = datetime.now().strftime("alert_%H%M%S.jpg")
                cv2.imwrite(os.path.join(snap_dir, name), frame)
                monitor.snapshot_pending = False

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                monitor.start_calibration(now)
            elif key == ord("m"):
                print("Alarm muted" if alarm.toggle_mute() else "Alarm unmuted")
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        mins, secs = divmod(int(time.monotonic() - session_start), 60)
        logger.event("SESSION_END",
                     details=f"duration {mins}m{secs:02d}s, alerts {monitor.alerts}, "
                             f"blinks {monitor.blinks}, yawns {monitor.yawns}")
        logger.close()
        print("\nSession summary")
        print(f"  Duration          : {mins}m {secs:02d}s")
        print(f"  Drowsiness alerts : {monitor.alerts}")
        print(f"  Blinks            : {monitor.blinks}")
        print(f"  Yawns             : {monitor.yawns}")
        print(f"  Event log         : {logger.path}")
        if os.path.isdir(snap_dir):
            print(f"  Alert snapshots   : {snap_dir}")


if __name__ == "__main__":
    main()
