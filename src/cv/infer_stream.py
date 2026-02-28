#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

def download_if_missing(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"[info] downloading MediaPipe model -> {dst}")
    urllib.request.urlretrieve(url, str(dst))
    return dst


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_float_key(s: str) -> Optional[float]:
    """Try parse vocab keys like '1.00' -> 1.0. Return None if not parseable."""
    try:
        return float(s)
    except Exception:
        return None


def normalize_pose_seq(L: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Produces per-frame features of length 198 using:
      xy (33*2) + vxy (33*2) + z (33*1) + vis (33*1) = 66+66+33+33 = 198

    Normalization:
      - translate by hip center (avg of landmarks 23 and 24)
      - scale by shoulder distance (||L11 - L12||)
    """
    L = np.nan_to_num(L, nan=0.0).astype(np.float32)
    V = np.nan_to_num(V, nan=0.0).astype(np.float32)


    hip = 0.5 * (L[:, 23, :2] + L[:, 24, :2])
    xy = L[:, :, :2] - hip[:, None, :]

    sh = L[:, 11, :2] - L[:, 12, :2]
    scale = np.linalg.norm(sh, axis=1)
    scale = np.clip(scale, 1e-3, None)
    xy = xy / scale[:, None, None]

    vxy = np.zeros_like(xy, dtype=np.float32)
    vxy[1:] = xy[1:] - xy[:-1]

    z = L[:, :, 2:3].astype(np.float32)
    vis = V[:, :, None].astype(np.float32)

    feat = np.concatenate([xy, vxy, z, vis], axis=2)
    feat = feat.reshape(L.shape[0], -1).astype(np.float32)
    return feat


class TCNBlock(nn.Module):
    """
    Causal TCN block: left-pad only, so output length == input length always.
    """
    def __init__(self, cin: int, cout: int, k: int, dilation: int, pdrop: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad1 = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv1 = nn.Conv1d(cin, cout, k, padding=0, dilation=dilation)
        self.pad2 = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=0, dilation=dilation)
        self.drop = nn.Dropout(pdrop)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.relu(self.conv1(self.pad1(x)))
        y = self.drop(y)
        y = F.relu(self.conv2(self.pad2(y)))
        y = self.drop(y)
        return y + self.res(x)


class PoseTCNTyped(nn.Module):
    def __init__(self, feat_dim, n_ex, n_mist, n_speed, hidden=256, layers=6):
        super().__init__()
        blocks = []
        cin = feat_dim
        for i in range(layers):
            blocks.append(TCNBlock(cin, hidden, k=3, dilation=2**i, pdrop=0.2))
            cin = hidden
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.ex_head = nn.Linear(hidden, n_ex)
        self.mist_head = nn.Linear(hidden, n_mist)
        self.speed_head = nn.Linear(hidden, n_speed)
        self.rom_head = nn.Linear(hidden, 5)
        self.height_head = nn.Linear(hidden, 5)
        self.torso_head = nn.Linear(hidden, 5)
        self.dir_head = nn.Linear(hidden, 3)
        self.no_issue_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)
        g = self.pool(h).squeeze(-1)
        return {
            "ex": self.ex_head(g),
            "mist": self.mist_head(g),
            "speed": self.speed_head(g),
            "rom": self.rom_head(g),
            "height": self.height_head(g),
            "torso": self.torso_head(g),
            "dir": self.dir_head(g),
            "no_issue": self.no_issue_head(g).squeeze(-1),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, or /path/to/video.mp4")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out-jsonl", type=str, default="")
    ap.add_argument("--mist-thresh", type=float, default=0.35)
    ap.add_argument("--min-exercise-p", type=float, default=0.25, help="If max(exercise) < this, label as 'unknown'")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    voc = ckpt.get("vocabs", {})
    ex2i: Dict[str, int] = voc.get("ex2i", {})
    mist2i: Dict[str, int] = voc.get("mist2i", {})
    speed2i: Dict[str, int] = voc.get("speed2i", {})

    rom2i: Dict[str, int] = voc.get("rom2i", {})
    height2i: Dict[str, int] = voc.get("height2i", {})
    torso2i: Dict[str, int] = voc.get("torso2i", {})
    dir2i: Dict[str, int] = voc.get("dir2i", {})

    i2ex = {i: k for k, i in ex2i.items()}
    i2mist = {i: k for k, i in mist2i.items()}
    i2speed = {i: k for k, i in speed2i.items()}

    i2rom = {i: k for k, i in rom2i.items()} if rom2i else {}
    i2height = {i: k for k, i in height2i.items()} if height2i else {}
    i2torso = {i: k for k, i in torso2i.items()} if torso2i else {}
    i2dir = {i: k for k, i in dir2i.items()} if dir2i else {0: "none", 1: "clockwise", 2: "counterclockwise"}

    feat_dim = int(ckpt.get("feat_dim", 198))
    if feat_dim != 198:
        raise RuntimeError(f"Checkpoint feat_dim={feat_dim} but infer_stream expects 198. (Did you change features?)")

    n_ex = max(1, len(ex2i))
    n_mist = max(1, len(mist2i))
    n_speed = max(1, len(speed2i))

    model = PoseTCNTyped(
        feat_dim=feat_dim,
        n_ex=n_ex,
        n_mist=n_mist,
        n_speed=n_speed,
        hidden=256,
        layers=6,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    device = pick_device(args.device)
    print("[device]", device)
    model.to(device)

    model_path = download_if_missing(MODEL_URL, Path("models") / "pose_landmarker_full.task")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    src: Union[int, str] = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    Lbuf: List[np.ndarray] = []
    Vbuf: List[np.ndarray] = []

    out_f = open(args.out_jsonl, "w") if args.out_jsonl else None
    t0 = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_bgr = ensure_bgr(frame)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)

            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                z = np.array([lm.z for lm in lms], dtype=np.float32)
                vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)
                L = np.concatenate([xy, z[:, None]], axis=1)
                V = vis
            else:
                L = np.full((33, 3), np.nan, dtype=np.float32)
                V = np.zeros((33,), dtype=np.float32)

            Lbuf.append(L)
            Vbuf.append(V)
            if len(Lbuf) > args.window:
                Lbuf.pop(0)
                Vbuf.pop(0)

            if len(Lbuf) == args.window and (frame_idx % args.stride == 0):
                Lseq = np.stack(Lbuf, axis=0)
                Vseq = np.stack(Vbuf, axis=0)
                X = normalize_pose_seq(Lseq, Vseq)

                with torch.no_grad():
                    xt = torch.from_numpy(X[None, ...]).to(device)
                    out = model(xt)

                    ex_prob = F.softmax(out["ex"], dim=1).cpu().numpy()[0]
                    mist_prob = torch.sigmoid(out["mist"]).cpu().numpy()[0]
                    no_issue_p = float(torch.sigmoid(out["no_issue"]).cpu().numpy()[0])

                    dir_prob = F.softmax(out["dir"], dim=1).cpu().numpy()[0]
                    speed_prob = (
                        F.softmax(out["speed"], dim=1).cpu().numpy()[0]
                        if out["speed"].shape[1] > 1 else None
                    )
                    rom_prob = F.softmax(out["rom"], dim=1).cpu().numpy()[0]
                    height_prob = F.softmax(out["height"], dim=1).cpu().numpy()[0]
                    torso_prob = F.softmax(out["torso"], dim=1).cpu().numpy()[0]

                ex_id = int(np.argmax(ex_prob))
                ex_p = float(ex_prob[ex_id])
                ex_name = i2ex.get(ex_id, str(ex_id))
                if ex_p < args.min_exercise_p:
                    ex_name = "unknown"
                exercise = {"name": ex_name, "p": ex_p}

                mistakes = []
                if mist_prob.size > 0:
                    top_idx = np.argsort(-mist_prob)[:10]
                    for i in top_idx:
                        p = float(mist_prob[int(i)])
                        if p >= args.mist_thresh:
                            mistakes.append({"name": i2mist.get(int(i), str(int(i))), "p": p})

                speed_val = None
                if speed_prob is not None and speed_prob.size > 0:
                    sid = int(np.argmax(speed_prob))
                    label = i2speed.get(sid, str(sid))
                    f = safe_float_key(label)
                    speed_val = f if f is not None else label

                rom_id = int(np.argmax(rom_prob))
                height_id = int(np.argmax(height_prob))
                torso_id = int(np.argmax(torso_prob))
                dir_id = int(np.argmax(dir_prob))

                rom_level = i2rom.get(rom_id, rom_id + 1)
                height_level = i2height.get(height_id, height_id + 1)
                torso_rotation = i2torso.get(torso_id, torso_id + 1)
                direction = i2dir.get(dir_id, {0: "none", 1: "clockwise", 2: "counterclockwise"}[dir_id])

                pen = float(np.sort(mist_prob)[-5:].mean()) if mist_prob.size else 0.0
                quality = float(np.clip(0.60 * no_issue_p + 0.40 * (1.0 - pen), 0.0, 1.0))

                event = {
                    "timestamp_s": float(time.perf_counter() - t0),
                    "frame_index": int(frame_idx),
                    "source_fps": float(fps),

                    "exercise": exercise,
                    "mistakes": mistakes,

                    "metrics": {
                        "speed_rps": speed_val,
                        "rom_level": rom_level,
                        "height_level": height_level,
                        "torso_rotation": torso_rotation,
                        "direction": direction,
                        "no_obvious_issue_p": no_issue_p,
                    },

                    "quality_score": quality,
                    "speak_now": 0.0,
                }

                s = json.dumps(event)
                print(s)
                if out_f:
                    out_f.write(s + "\n")
                    out_f.flush()

            if args.show:
                cv2.imshow("infer_stream", frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            frame_idx += 1

    finally:
        cap.release()
        landmarker.close()
        if out_f:
            out_f.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
