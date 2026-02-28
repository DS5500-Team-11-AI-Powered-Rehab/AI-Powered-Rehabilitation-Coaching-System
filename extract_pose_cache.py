#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp_mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


SKIP_PART1_FIRST8 = {f"{i:08d}.mp4" for i in range(8)}

_G_LANDMARKER = None
_G_ARGS = None


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


def build_video_index(parts: List[Path]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in parts:
        for mp4_ in p.glob("*.mp4"):
            idx[mp4_.name] = mp4_.resolve()
    if not idx:
        raise RuntimeError("No .mp4 files found in provided part directories.")
    return idx


def detect_pose_image(landmarker: vision.PoseLandmarker, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks[0]  # 33
    xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
    z = np.array([lm.z for lm in lms], dtype=np.float32)
    vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)
    xyz = np.concatenate([xy, z[:, None]], axis=1)  # (33,3)
    return xyz, vis


def compute_stability_start(
    L: np.ndarray, V: np.ndarray, fps: float,
    *,
    vis_thresh: float = 0.5,
    good_frac_thresh: float = 0.65,
    stable_window: int = 10,
    jitter_thresh: float = 0.035,
    max_trim_seconds: float = 2.0,
) -> Tuple[Optional[int], str]:
    T = L.shape[0]
    if T < stable_window + 1:
        return None, "too_short"

    max_trim = int(max_trim_seconds * fps)
    max_start = min(T - stable_window - 1, max_trim)

    good_frac = (V >= vis_thresh).mean(axis=1)  # (T,)

    xy = np.nan_to_num(L[:, :, :2], nan=0.0)
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=2)  # (T-1,33)
    jitter = np.median(d, axis=1)                 # (T-1,)
    jitter = np.concatenate([[jitter[0]], jitter])  # align to (T,)

    for t in range(0, max_start + 1):
        gf = good_frac[t:t + stable_window]
        jj = jitter[t:t + stable_window]
        if (gf >= good_frac_thresh).all() and (jj <= jitter_thresh).all():
            return t, "stable_found"

    return None, "no_stable_segment_in_early_window"


def save_debug_overlay_png(out_png: Path, frame_bgr: np.ndarray, xyz: np.ndarray, vis: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    pts = (xyz[:, :2] * np.array([w, h], dtype=np.float32)).astype(int)

    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 31), (28, 32),
    ]
    for a, b in edges:
        if vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for j in range(33):
        if vis[j] > 0.3:
            cv2.circle(out, tuple(pts[j]), 3, (0, 255, 0), -1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), out)


def _worker_init(model_path_str: str, args_dict: dict):
    global _G_LANDMARKER, _G_ARGS

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    base_options = python.BaseOptions(model_asset_path=model_path_str)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    _G_LANDMARKER = vision.PoseLandmarker.create_from_options(options)
    _G_ARGS = args_dict


def _process_one(task: dict) -> Tuple[str, Counter]:
    global _G_LANDMARKER, _G_ARGS
    stats = Counter()

    in_path = Path(task["in_path"])
    out_npz = Path(task["out_npz"])

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        stats["skip_cant_open"] += 1
        return in_path.stem, stats

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    L_list = []
    V_list = []
    first_good_overlay = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = ensure_bgr(frame)
            det = detect_pose_image(_G_LANDMARKER, frame)
            if det is None:
                L_list.append(np.full((33, 3), np.nan, dtype=np.float32))
                V_list.append(np.zeros((33,), dtype=np.float32))
            else:
                xyz, vis = det
                L_list.append(xyz)
                V_list.append(vis)
                if first_good_overlay is None and task.get("do_debug", False) and (vis >= 0.5).mean() >= 0.65:
                    first_good_overlay = (frame.copy(), xyz.copy(), vis.copy())
    finally:
        cap.release()

    if not L_list:
        stats["skip_empty_video"] += 1
        return in_path.stem, stats

    L = np.stack(L_list, axis=0)
    V = np.stack(V_list, axis=0)

    start_idx, reason = compute_stability_start(
        L, V, fps,
        vis_thresh=float(_G_ARGS["vis_thresh"]),
        good_frac_thresh=float(_G_ARGS["good_frac_thresh"]),
        stable_window=int(_G_ARGS["stable_window"]),
        jitter_thresh=float(_G_ARGS["jitter_thresh"]),
        max_trim_seconds=float(_G_ARGS["max_trim_seconds"]),
    )

    if start_idx is None:
        stats[f"unstable_{reason}"] += 1
        if bool(_G_ARGS["skip_if_unstable"]):
            return in_path.stem, stats
        start_idx = 0
        reason = f"fallback_keep_all_{reason}"

    L2 = L[start_idx:]
    V2 = V[start_idx:]

    np.savez_compressed(
        out_npz,
        landmarks=L2.astype(np.float32),
        visibility=V2.astype(np.float32),
        fps=np.float32(fps),
        split=str(task.get("split", "train")),
        labels=np.array([str(s) for s in task["labels"]], dtype=object),
        src=str(in_path),
        trim_start=np.int32(start_idx),
        trim_reason=str(reason),
    )
    stats["written"] += 1

    if task.get("do_debug", False) and first_good_overlay is not None and task.get("dbg_png"):
        fr, xyz, vis = first_good_overlay
        save_debug_overlay_png(Path(task["dbg_png"]), fr, xyz, vis)

    return in_path.stem, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-json", type=str, required=True)
    ap.add_argument("--parts", type=str, nargs="+", required=True)
    ap.add_argument("--out-dir", type=str, default="pose_cache")
    ap.add_argument("--debug-dir", type=str, default="pose_cache_debug")
    ap.add_argument("--debug-n", type=int, default=10)
    ap.add_argument("--max-items", type=int, default=0, help="0=all")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--workers", type=int, default=10,
                    help="Number of parallel worker processes. For M4 Max 12P+4E, try 8-12 (default 10).")

    ap.add_argument("--vis-thresh", type=float, default=0.5)
    ap.add_argument("--good-frac-thresh", type=float, default=0.65)
    ap.add_argument("--stable-window", type=int, default=10)
    ap.add_argument("--jitter-thresh", type=float, default=0.035)
    ap.add_argument("--max-trim-seconds", type=float, default=2.0)
    ap.add_argument("--skip-if-unstable", action="store_true", help="skip clip if stable segment not found")
    args = ap.parse_args()

    labels_path = Path(args.labels_json).expanduser().resolve()
    parts = [Path(p).expanduser().resolve() for p in args.parts]
    out_dir = Path(args.out_dir).expanduser().resolve()
    dbg_dir = Path(args.debug_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open("r") as f:
        items = json.load(f)

    video_index = build_video_index(parts)

    model_path = download_if_missing(MODEL_URL, Path("models") / "pose_landmarker_full.task")

    n_total = len(items) if args.max_items <= 0 else min(len(items), args.max_items)

    stats = Counter()
    tasks = []
    dbg_left = int(args.debug_n)

    for it in items[:n_total]:
        labels = it.get("labels", [])
        if not labels:
            stats["skip_missing_labels"] += 1
            continue

        vrel = str(it["video_path"]).replace("./", "").strip()
        fname = Path(vrel).name

        if fname in SKIP_PART1_FIRST8:
            stats["skip_part1_first8"] += 1
            continue

        if fname not in video_index:
            stats["skip_not_found"] += 1
            continue

        in_path = video_index[fname]
        out_npz = out_dir / f"{in_path.stem}.npz"
        if out_npz.exists() and not args.overwrite:
            stats["skip_exists"] += 1
            continue

        do_debug = dbg_left > 0
        dbg_png = str(dbg_dir / f"{in_path.stem}_overlay.png") if do_debug else ""
        if do_debug:
            dbg_left -= 1

        tasks.append({
            "in_path": str(in_path),
            "out_npz": str(out_npz),
            "labels": [str(s) for s in labels],
            "split": str(it.get("split", "train")),
            "do_debug": do_debug,
            "dbg_png": dbg_png,
        })

    args_dict = dict(
        vis_thresh=args.vis_thresh,
        good_frac_thresh=args.good_frac_thresh,
        stable_window=args.stable_window,
        jitter_thresh=args.jitter_thresh,
        max_trim_seconds=args.max_trim_seconds,
        skip_if_unstable=bool(args.skip_if_unstable),
    )

    mp_ctx = mp_mp.get_context("spawn")

    t0 = time.perf_counter()

    n_workers = max(1, int(args.workers))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_ctx,
        initializer=_worker_init,
        initargs=(str(model_path), args_dict),
    ) as ex:

        futures = [ex.submit(_process_one, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="pose_cache_clean_parallel"):
            try:
                _stem, st = fut.result()
                stats.update(st)
            except Exception:
                stats["error_exception"] += 1

    print("\n[summary]")
    for k, v in stats.most_common():
        print(f"{k:28s} {v}")
    print(f"[done] total_time_s={time.perf_counter() - t0:.1f} out_dir={out_dir}")


if __name__ == "__main__":
    main()
