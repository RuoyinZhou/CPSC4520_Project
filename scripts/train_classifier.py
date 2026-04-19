#!/usr/bin/env python
"""Train resnet1d_wang on PTB-XL 5-superclass multi-label task.

Inputs: 10 s 12-lead records at 100 Hz (1000 samples), official strat_fold
splits (1-8 train, 9 val, 10 test). Loss: BCE-with-logits + class weights
(inverse-prevalence, normalized to mean 1). Success: macro-AUC >= 0.92 on
fold 10.

Reports per-class + macro AUC on fold 10 and saves the best (val-macro-AUC)
checkpoint to results/classifier/best.pt.
"""
from __future__ import annotations
import argparse, ast, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model_resnet1d_wang import ResNet1dWang  # noqa: E402

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_scp_to_super(scp_csv: Path) -> dict[str, str]:
    df = pd.read_csv(scp_csv, index_col=0)
    df = df[df.diagnostic == 1.0]
    return {k: v for k, v in df.diagnostic_class.items() if v in SUPERCLASSES}


class PTBXLDataset(Dataset):
    """100 Hz records, 12-lead, 10 s = 1000 samples; preloaded in memory."""

    def __init__(self, db: pd.DataFrame, root: Path, scp_map: dict[str, str],
                 cache_path: Path | None = None):
        self.db = db.reset_index()
        self.root = root
        self.scp_map = scp_map
        self.X = None
        self.Y = None
        if cache_path is not None and cache_path.exists():
            print(f"[cache] loading {cache_path}", flush=True)
            with h5py_file(cache_path, "r") as f:
                self.X = np.asarray(f["X"])
                self.Y = np.asarray(f["Y"])

    def _y(self, scp_codes):
        if isinstance(scp_codes, str):
            scp_codes = ast.literal_eval(scp_codes)
        sc = {self.scp_map[c] for c in scp_codes if c in self.scp_map}
        return np.array([float(s in sc) for s in SUPERCLASSES], dtype=np.float32)

    def preload(self):
        X = np.zeros((len(self.db), 12, 1000), dtype=np.float32)
        Y = np.zeros((len(self.db), 5), dtype=np.float32)
        for i, row in enumerate(self.db.itertuples()):
            sig, _ = wfdb.rdsamp(str(self.root / row.filename_lr))
            x = sig.astype(np.float32).T
            if x.shape[1] != 1000:
                pad = 1000 - x.shape[1]
                x = np.pad(x, ((0, 0), (0, max(0, pad))))[:, :1000]
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
            X[i] = x
            Y[i] = self._y(row.scp_codes)
            if (i + 1) % 2000 == 0:
                print(f"  [preload] {i+1}/{len(self.db)}", flush=True)
        self.X = X; self.Y = Y

    def save_cache(self, path: Path):
        import h5py as _h5
        with _h5.File(path, "w") as f:
            f.create_dataset("X", data=self.X, compression="gzip", compression_opts=4)
            f.create_dataset("Y", data=self.Y)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def h5py_file(path, mode):
    import h5py
    return h5py.File(path, mode)


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    per = {}
    aucs = []
    for i, c in enumerate(SUPERCLASSES):
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == y_true.shape[0]:
            per[c] = float("nan")
            continue
        auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        per[c] = auc
        aucs.append(auc)
    return (float(np.mean(aucs)) if aucs else float("nan")), per


def run_epoch(model, loader, crit, opt, device, train: bool, sched_batch=None):
    model.train(train)
    tot, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = crit(logits, y)
            if train:
                loss.backward()
                opt.step()
                if sched_batch is not None:
                    sched_batch.step()
        bs = x.size(0)
        tot += loss.item() * bs; n += bs
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(ys); y_score = np.concatenate(ps)
    m_auc, per = macro_auc(y_true, y_score)
    return tot / n, m_auc, per, y_true, y_score


def main(args):
    root = Path(args.ptbxl_root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db = db[db.strat_fold.notna()].copy()
    scp_map = load_scp_to_super(root / "scp_statements.csv")

    train_db = db[db.strat_fold.isin(range(1, 9))]
    val_db = db[db.strat_fold == 9]
    test_db = db[db.strat_fold == 10]
    if args.quick:
        train_db = train_db.head(512); val_db = val_db.head(128); test_db = test_db.head(128)
    print(f"train={len(train_db)} val={len(val_db)} test={len(test_db)}", flush=True)

    cache_dir = Path(args.out_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = PTBXLDataset(train_db, root, scp_map, cache_dir / "cache_train.h5")
    val_ds = PTBXLDataset(val_db, root, scp_map, cache_dir / "cache_val.h5")
    test_ds = PTBXLDataset(test_db, root, scp_map, cache_dir / "cache_test.h5")
    for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        if ds.X is None:
            print(f"[preload] {name}", flush=True)
            ds.preload()
            ds.save_cache(cache_dir / f"cache_{name}.h5")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    # class weights (inverse prevalence, normalized mean 1)
    Y = train_ds.Y.copy()
    pos = Y.sum(axis=0); neg = (1 - Y).sum(axis=0)
    pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32)
    pos_weight = pos_weight * (5.0 / pos_weight.sum())
    print("pos_weight:", dict(zip(SUPERCLASSES, pos_weight.tolist())), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet1dWang(num_classes=5, input_channels=12).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Scheduler setup: 'plateau' (our default) or 'onecycle' (Strodthoff benchmark)
    sched_batch = None
    if args.scheduler == "onecycle":
        # Match Strodthoff benchmark: AdamW + OneCycleLR (fastai fit_one_cycle)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched_batch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_dl))
        sched_epoch = None
        print(f"[scheduler] OneCycleLR: max_lr={args.lr}, wd={args.wd}, "
              f"steps/epoch={len(train_dl)}, total_steps={args.epochs*len(train_dl)}", flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched_epoch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=args.patience, min_lr=1e-4)
        print(f"[scheduler] ReduceLROnPlateau: patience={args.patience}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0; best_path = out_dir / "best.pt"
    best_epoch = 0
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_macro_auc," + ",".join(f"val_{c}" for c in SUPERCLASSES) + ",lr\n")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, _, _, _, _ = run_epoch(model, train_dl, crit, opt, device, True, sched_batch=sched_batch)
        vl, va, vper, _, _ = run_epoch(model, val_dl, crit, opt, device, False)
        if sched_epoch is not None:
            sched_epoch.step(va)
        cur_lr = opt.param_groups[0]['lr']
        with open(log_path, "a") as f:
            f.write(f"{ep},{tl:.4f},{vl:.4f},{va:.4f}," + ",".join(f"{vper[c]:.4f}" for c in SUPERCLASSES) + f",{cur_lr:.2e}\n")
        print(f"[ep{ep:03d}] train={tl:.4f} val={vl:.4f} macro-AUC={va:.4f} "
              f"per={vper} lr={cur_lr:.2e} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        if va > best_val:
            best_val = va
            best_epoch = ep
            torch.save({"model": model.state_dict(), "val_macro_auc": va,
                        "per_class": vper, "epoch": ep}, best_path)
        # early stopping (disabled for onecycle which runs fixed epochs)
        if args.early_stop > 0 and args.scheduler != "onecycle" and (ep - best_epoch) >= args.early_stop:
            print(f"[early stop] no improvement for {args.early_stop} epochs (best@ep{best_epoch})", flush=True)
            break

    # final test eval with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, ta, tper, yt, ys = run_epoch(model, test_dl, crit, opt, device, False)
    print(f"[TEST best-val@ep{ckpt['epoch']}] macro-AUC={ta:.4f} per={tper}", flush=True)
    np.savez(out_dir / "test_predictions.npz",
             y_true=yt, y_score=ys, classes=np.array(SUPERCLASSES))
    with open(out_dir / "test_metrics.txt", "w") as f:
        f.write(f"macro_auc {ta:.6f}\n")
        for c in SUPERCLASSES:
            f.write(f"{c}_auc {tper[c]:.6f}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=7, help="ReduceLROnPlateau patience")
    p.add_argument("--early_stop", type=int, default=150, help="stop after N epochs without improvement (0=disable)")
    p.add_argument("--scheduler", choices=["plateau", "onecycle"], default="plateau",
                   help="LR scheduler: plateau (Adam+ReduceLR) or onecycle (AdamW+OneCycleLR, matches Strodthoff)")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay (for AdamW with onecycle)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="smoke test with a few hundred records")
    main(p.parse_args())
