#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TCNBlock(nn.Module):
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
        blocks=[]
        cin=feat_dim
        for i in range(layers):
            blocks.append(TCNBlock(cin, hidden, k=3, dilation=2**i, pdrop=0.2))
            cin=hidden
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
        x = x.transpose(1,2)
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

class StepETAMeter:
    def __init__(self, total_steps:int):
        self.total=max(1,int(total_steps))
        self.done=0
        self.t0=time.perf_counter()
    def update(self,n=1): self.done+=int(n)
    def pretty(self):
        elapsed=time.perf_counter()-self.t0
        rate=self.done/elapsed if elapsed>0 else 0
        left=max(0,self.total-self.done)
        eta=left/rate if rate>0 else float("inf")
        if not math.isfinite(eta): return "ETA=??"
        m,s=divmod(int(eta),60); h,m=divmod(m,60)
        return f"ETA={h:02d}:{m:02d}:{s:02d}"

def pick_device(choice:str)->torch.device:
    if choice!="auto":
        return torch.device(choice)
    if getattr(torch.backends,"mps",None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_splits(exercise_id: np.ndarray, seed:int, val_frac:float=0.08):
    rng=random.Random(seed)
    by_ex={}
    for i,e in enumerate(exercise_id.tolist()):
        by_ex.setdefault(e,[]).append(i)
    for e in by_ex: rng.shuffle(by_ex[e])

    eligible=[e for e,lst in by_ex.items() if len(lst)>=1]
    test=[]
    remain=[]
    for e,lst in by_ex.items():
        if e in eligible:
            test += lst[:2]
            remain += lst[2:]
        else:
            remain += lst

    remain_by={}
    for i in remain:
        e=int(exercise_id[i])
        remain_by.setdefault(e,[]).append(i)

    train=[]
    val=[]
    for e,lst in remain_by.items():
        rng.shuffle(lst)
        n_val=int(round(len(lst)*val_frac))
        if len(lst)>=2:
            n_val=max(1,n_val)
            n_val=min(n_val,len(lst)-1)
        else:
            n_val=0
        val += lst[:n_val]
        train += lst[n_val:]

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return np.array(train,dtype=np.int64), np.array(val,dtype=np.int64), np.array(test,dtype=np.int64), len(by_ex), len(eligible)

class MemmapWindowDataset(Dataset):
    def __init__(self, X_mmap, offsets, lengths, idxs,
                 exercise_id, mist, speed_id, rom_id, height_id, torso_id, dir_id, no_issue,
                 window:int, seed:int):
        self.X = X_mmap
        self.offsets=offsets
        self.lengths=lengths
        self.idxs=idxs
        self.exercise_id=exercise_id
        self.mist=mist
        self.speed_id=speed_id
        self.rom_id=rom_id
        self.height_id=height_id
        self.torso_id=torso_id
        self.dir_id=dir_id
        self.no_issue=no_issue
        self.window=int(window)
        self.rng=random.Random(seed)

    def __len__(self): return int(self.idxs.size)

    def __getitem__(self, k):
        i = int(self.idxs[k])
        T = int(self.lengths[i])
        o = int(self.offsets[i])

        if T >= self.window:
            start = self.rng.randint(0, T - self.window)
            x = np.array(self.X[o+start:o+start+self.window], dtype=np.float32)  # convert f16->f32
        else:
            x = np.zeros((self.window, 198), dtype=np.float32)
            x[:T] = np.array(self.X[o:o+T], dtype=np.float32)

        return (
            torch.from_numpy(x),
            torch.tensor(int(self.exercise_id[i]), dtype=torch.long),
            torch.from_numpy(self.mist[i].astype(np.float32)),
            torch.tensor(int(self.speed_id[i]), dtype=torch.long),
            torch.tensor(int(self.rom_id[i]), dtype=torch.long),
            torch.tensor(int(self.height_id[i]), dtype=torch.long),
            torch.tensor(int(self.torso_id[i]), dtype=torch.long),
            torch.tensor(int(self.dir_id[i]), dtype=torch.long),
            torch.tensor(float(self.no_issue[i]), dtype=torch.float32),
        )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--run-name", type=str, default="qevd_full_memmap")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    args=ap.parse_args()

    data_dir=Path(args.data_dir).expanduser().resolve()
    voc=json.loads((data_dir/"vocabs.json").read_text())
    n_ex=len(voc["ex2i"])
    n_mist=len(voc["mist2i"])
    n_speed=max(1, len(voc["speed2i"]))

    offsets=np.load(data_dir/"offsets.npy")
    lengths=np.load(data_dir/"lengths.npy")
    exercise_id=np.load(data_dir/"exercise_id.npy")
    mist=np.load(data_dir/"mist.npy")
    speed_id=np.load(data_dir/"speed_id.npy")
    rom_id=np.load(data_dir/"rom_id.npy")
    height_id=np.load(data_dir/"height_id.npy")
    torso_id=np.load(data_dir/"torso_id.npy")
    dir_id=np.load(data_dir/"dir_id.npy")
    no_issue=np.load(data_dir/"no_issue.npy")

    X = np.memmap(data_dir/"X.f16.mmap", dtype=np.float16, mode="r", shape=(int(lengths.sum()), 198))

    train_idx, val_idx, test_idx, n_exercises_total, n_eligible = build_splits(exercise_id, args.seed, args.val_frac)
    print(f"[split] exercises_total={n_exercises_total} eligible_for_test(count>=8)={n_eligible}")
    print(f"[split] sizes: train={train_idx.size} val={val_idx.size} test={test_idx.size}")

    train_ds=MemmapWindowDataset(X, offsets, lengths, train_idx, exercise_id, mist, speed_id, rom_id, height_id, torso_id, dir_id, no_issue, args.window, args.seed)
    val_ds=MemmapWindowDataset(X, offsets, lengths, val_idx, exercise_id, mist, speed_id, rom_id, height_id, torso_id, dir_id, no_issue, args.window, args.seed+1)
    test_ds=MemmapWindowDataset(X, offsets, lengths, test_idx, exercise_id, mist, speed_id, rom_id, height_id, torso_id, dir_id, no_issue, args.window, args.seed+2)

    persistent = args.num_workers > 0
    train_dl=DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        persistent_workers=persistent, prefetch_factor=4 if persistent else None)
    val_dl=DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False,
                      persistent_workers=persistent, prefetch_factor=4 if persistent else None)
    test_dl=DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False,
                       persistent_workers=persistent, prefetch_factor=4 if persistent else None)

    model=PoseTCNTyped(198, n_ex, n_mist, n_speed, hidden=256, layers=6)
    device=pick_device(args.device)
    print("[device]", device)
    model.to(device)

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_mist=nn.BCEWithLogitsLoss()
    bce_bin=nn.BCEWithLogitsLoss()
    ce_ignore=nn.CrossEntropyLoss(ignore_index=-1)

    ckpt_dir=Path("checkpoints")/args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val=-1.0

    def eval_ex_acc(dl):
        model.eval()
        c=0;t=0
        with torch.no_grad():
            for Xb, y_ex, *_ in dl:
                Xb=Xb.to(device)
                y_ex=y_ex.to(device)
                out=model(Xb)
                pred=torch.argmax(out["ex"], dim=1)
                c += int((pred==y_ex).sum().item())
                t += int(y_ex.numel())
        return c/max(1,t)

    total_steps=args.epochs*len(train_dl)
    meter=StepETAMeter(total_steps)
    step=0

    for ep in range(1, args.epochs+1):
        model.train()
        losses=[]
        for batch in train_dl:
            Xb, y_ex, y_mist, y_speed, y_rom, y_h, y_t, y_dir, y_no = batch
            Xb=Xb.to(device)
            y_ex=y_ex.to(device)
            y_mist=y_mist.to(device)
            y_speed=y_speed.to(device)
            y_rom=y_rom.to(device)
            y_h=y_h.to(device)
            y_t=y_t.to(device)
            y_dir=y_dir.to(device)
            y_no=y_no.to(device)

            out=model(Xb)

            loss_ex=F.cross_entropy(out["ex"], y_ex)
            loss_m= bce_mist(out["mist"], y_mist)
            loss_speed= ce_ignore(out["speed"], y_speed) if out["speed"].shape[1]>1 else 0.0
            loss_rom= ce_ignore(out["rom"], y_rom)
            loss_h= ce_ignore(out["height"], y_h)
            loss_t= ce_ignore(out["torso"], y_t)
            loss_dir=F.cross_entropy(out["dir"], y_dir)
            loss_no=bce_bin(out["no_issue"], y_no)

            loss = 1.0*loss_ex + 0.8*loss_m + 0.3*loss_speed + 0.3*loss_rom + 0.2*loss_h + 0.2*loss_t + 0.2*loss_dir + 0.2*loss_no

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))
            step += 1
            meter.update(1)
            if step % 50 == 0:
                print(f"[ep {ep}/{args.epochs}] step {step}/{total_steps} loss={np.mean(losses):.4f} {meter.pretty()}")

        val_acc=eval_ex_acc(val_dl)
        test_acc=eval_ex_acc(test_dl)
        print(f"[eval] epoch={ep} val_ex_acc={val_acc:.4f} test_ex_acc={test_acc:.4f}")

        ckpt={"epoch":ep,"model":model.state_dict(),"opt":opt.state_dict(),"vocabs":voc,
              "feat_dim":198,"window":args.window,"val_acc":val_acc,"test_acc":test_acc}


        if val_acc>best_val + 0.1 and ep>100:
            best_val=val_acc
            torch.save(ckpt, ckpt_dir/"best.pt")
            print(f"[ckpt] new best val={best_val:.4f}")

        if args.save_every>0 and ep%args.save_every==0:
            torch.save(ckpt, ckpt_dir/f"epoch_{ep:04d}.pt")
            print(f"[ckpt] saved epoch_{ep:04d}.pt")

    print("[done] ckpts in", ckpt_dir)

if __name__=="__main__":
    main()
