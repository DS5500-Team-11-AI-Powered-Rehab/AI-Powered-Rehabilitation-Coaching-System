#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

RE_SPEED = re.compile(r"\bspeed\s*=\s*([0-9]*[.]?[0-9]+)\s*rps\b", re.IGNORECASE)
RE_ROM = re.compile(r"\brom\s*=\s*([0-9]+)\b", re.IGNORECASE)
RE_HEIGHT = re.compile(r"\bheight\s*=\s*([0-9]+)\b", re.IGNORECASE)
RE_TORSO_ROT = re.compile(r"\btorso_rotation\s*=\s*([0-9]+)\b", re.IGNORECASE)
RE_CLOCKWISE = re.compile(r"\bclockwise\b", re.IGNORECASE)
RE_COUNTERCLOCKWISE = re.compile(r"\bcounter\s*clockwise\b|\bcounterclockwise\b", re.IGNORECASE)

def split_label(lbl: str) -> Tuple[str, str]:
    if " - " in lbl:
        ex, attr = lbl.split(" - ", 1)
        return ex.strip(), attr.strip()
    return lbl.strip(), ""

def infer_attr_type(attr: str) -> str:
    a = attr.strip().lower()
    if RE_SPEED.search(a): return "speed"
    if RE_ROM.search(a): return "rom"
    if RE_HEIGHT.search(a): return "height"
    if RE_TORSO_ROT.search(a): return "torso_rotation"
    if RE_COUNTERCLOCKWISE.search(a) or RE_CLOCKWISE.search(a): return "direction"
    if a == "no obvious issue": return "no_issue"
    return "other"

def normalize_pose_seq(L: np.ndarray, V: np.ndarray) -> np.ndarray:
    L = np.nan_to_num(L, nan=0.0).astype(np.float32)
    V = np.nan_to_num(V, nan=0.0).astype(np.float32)

    hip = 0.5 * (L[:, 23, :2] + L[:, 24, :2])
    xy = L[:, :, :2] - hip[:, None, :]

    sh = L[:, 11, :2] - L[:, 12, :2]
    scale = np.linalg.norm(sh, axis=1)
    scale = np.clip(scale, 1e-3, None)
    xy = xy / scale[:, None, None]

    z = L[:, :, 2:3]
    vis = V[:, :, None]

    vxy = np.zeros_like(xy)
    vxy[1:] = xy[1:] - xy[:-1]

    feat = np.concatenate([xy, vxy, z, vis], axis=2)  # (T,33,6)
    return feat.reshape(feat.shape[0], -1).astype(np.float32)  # (T,198)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-cache", type=str, nargs="+", required=True)
    ap.add_argument("--out-dir", type=str, default="qevd_memmap")
    ap.add_argument("--max-files", type=int, default=0, help="0=all (for quick tests)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather npz paths
    npz_paths: List[Path] = []
    for d in args.pose_cache:
        p = Path(d).expanduser().resolve()
        npz_paths.extend(sorted(p.rglob("*.npz")))
    npz_paths = sorted(npz_paths)
    if args.max_files > 0:
        npz_paths = npz_paths[:args.max_files]
    if not npz_paths:
        raise RuntimeError("No npz files found. Check --pose-cache paths.")

    print(f"[info] npz files: {len(npz_paths)}")

    exercises = set()
    other_attrs = set()
    speed_vals = set()

    lengths = np.zeros((len(npz_paths),), dtype=np.int32)

    for i, p in enumerate(npz_paths):
        z = np.load(p, allow_pickle=True)
        labels = [str(s) for s in z["labels"]]
        if not labels:
            lengths[i] = 0
            continue

        ex, _ = split_label(labels[0])
        exercises.add(ex)
        T = int(z["landmarks"].shape[0])
        lengths[i] = T

        for s in labels:
            _, attr = split_label(s)
            if not attr:
                continue
            atype = infer_attr_type(attr)
            if atype == "other":
                other_attrs.add(attr)
            elif atype == "speed":
                m = RE_SPEED.search(attr)
                if m:
                    speed_vals.add(m.group(1))

    keep_mask = lengths > 0
    npz_paths = [p for p, k in zip(npz_paths, keep_mask) if k]
    lengths = lengths[keep_mask]

    N = len(npz_paths)
    total_frames = int(lengths.sum())
    print(f"[info] usable clips: {N}")
    print(f"[info] total_frames: {total_frames}")

    ex2i = {k: i for i, k in enumerate(sorted(exercises))}
    mist2i = {k: i for i, k in enumerate(sorted(other_attrs))}
    speed2i = {k: i for i, k in enumerate(sorted(speed_vals, key=lambda x: float(x)))}

    vocabs = {
        "ex2i": ex2i,
        "mist2i": mist2i,
        "speed2i": speed2i,
        "rom_values": [1,2,3,4,5],
        "height_values": [1,2,3,4,5],
        "torso_values": [1,2,3,4,5],
        "dir_values": ["none","clockwise","counterclockwise"],
        "feat_dim": 198,
    }
    (out_dir / "vocabs.json").write_text(json.dumps(vocabs))

    offsets = np.zeros((N,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])

    X_path = out_dir / "X.f16.mmap"
    X = np.memmap(X_path, dtype=np.float16, mode="w+", shape=(total_frames, 198))

    exercise_id = np.zeros((N,), dtype=np.int32)
    mist = np.zeros((N, len(mist2i)), dtype=np.uint8)
    speed_id = np.full((N,), -1, dtype=np.int16)
    rom_id = np.full((N,), -1, dtype=np.int16)
    height_id = np.full((N,), -1, dtype=np.int16)
    torso_id = np.full((N,), -1, dtype=np.int16)
    dir_id = np.zeros((N,), dtype=np.int8)
    no_issue = np.zeros((N,), dtype=np.float16)

    for i, p in enumerate(npz_paths):
        z = np.load(p, allow_pickle=True)
        labels = [str(s) for s in z["labels"]]
        ex, _ = split_label(labels[0])
        exercise_id[i] = ex2i[ex]

        for s in labels:
            _, attr = split_label(s)
            if not attr:
                continue
            atype = infer_attr_type(attr)
            if atype == "no_issue":
                no_issue[i] = np.float16(1.0)
            elif atype == "direction":
                if RE_COUNTERCLOCKWISE.search(attr):
                    dir_id[i] = 2
                elif RE_CLOCKWISE.search(attr):
                    dir_id[i] = 1
            elif atype == "speed":
                m = RE_SPEED.search(attr)
                if m and m.group(1) in speed2i:
                    speed_id[i] = np.int16(speed2i[m.group(1)])
            elif atype == "rom":
                m = RE_ROM.search(attr)
                if m:
                    k = int(m.group(1))
                    if 1 <= k <= 5:
                        rom_id[i] = np.int16(k-1)
            elif atype == "height":
                m = RE_HEIGHT.search(attr)
                if m:
                    k = int(m.group(1))
                    if 1 <= k <= 5:
                        height_id[i] = np.int16(k-1)
            elif atype == "torso_rotation":
                m = RE_TORSO_ROT.search(attr)
                if m:
                    k = int(m.group(1))
                    if 1 <= k <= 5:
                        torso_id[i] = np.int16(k-1)
            else:
                j = mist2i.get(attr, None)
                if j is not None:
                    mist[i, j] = 1

        L = z["landmarks"].astype(np.float32)
        V = z["visibility"].astype(np.float32)
        feat = normalize_pose_seq(L, V)
        o = int(offsets[i])
        T = int(lengths[i])
        X[o:o+T] = feat.astype(np.float16)

        if (i+1) % 2000 == 0:
            print(f"[write] {i+1}/{N}")

    X.flush()

    np.save(out_dir / "offsets.npy", offsets)
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "exercise_id.npy", exercise_id)
    np.save(out_dir / "mist.npy", mist)
    np.save(out_dir / "speed_id.npy", speed_id)
    np.save(out_dir / "rom_id.npy", rom_id)
    np.save(out_dir / "height_id.npy", height_id)
    np.save(out_dir / "torso_id.npy", torso_id)
    np.save(out_dir / "dir_id.npy", dir_id)
    np.save(out_dir / "no_issue.npy", no_issue)

    print(f"[done] wrote memmap dataset to {out_dir}")

if __name__ == "__main__":
    main()
