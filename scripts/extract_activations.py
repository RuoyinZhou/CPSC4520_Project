#!/usr/bin/env python
"""Fast extraction of per-beat stage1/2/3 activations from resnet1d_wang.

v3 optimisations over v2:
  * No h5 compression (writes ~40-50x faster; disk is cheap on beegfs).
  * Flush in whole-chunk slabs, not per-beat slices.
  * Single joblib threading pool reused across batches.
  * Keep 1-record forward pass (batching across records gave no wins —
    the bottleneck was I/O, not matmul).

Output layout unchanged: {stage}_activations.h5 with float16 values.
"""
from __future__ import annotations
import argparse, ast, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import wfdb
import neurokit2 as nk
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from model_resnet1d_wang import ResNet1dWang  # noqa: E402

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 100
HALF_WIN_S = 0.4
HALF_100 = int(FS * HALF_WIN_S)
STAGE_DOWN = {"stage1": 1, "stage2": 2, "stage3": 4}
STAGE_HALF = {s: max(1, HALF_100 // d) for s, d in STAGE_DOWN.items()}


def load_scp_to_super(scp_csv):
    df = pd.read_csv(scp_csv, index_col=0)
    df = df[df.diagnostic == 1.0]
    return {k: v for k, v in df.diagnostic_class.items() if v in SUPERCLASSES}


def y_vec(scp_codes, scp_map):
    if isinstance(scp_codes, str):
        scp_codes = ast.literal_eval(scp_codes)
    sc = {scp_map[c] for c in scp_codes if c in scp_map}
    return np.array([float(s in sc) for s in SUPERCLASSES], dtype=np.float32)


def _load_record(path_str):
    sig, _ = wfdb.rdsamp(path_str)
    x = sig.astype(np.float32).T
    if x.shape[1] != 1000:
        pad = 1000 - x.shape[1]
        x = np.pad(x, ((0, 0), (0, max(0, pad))))[:, :1000]
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    try:
        _, info = nk.ecg_peaks(x[1], sampling_rate=FS, method="neurokit")
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int32)
    except Exception:
        rpeaks = np.array([], dtype=np.int32)
    return x, rpeaks


def main(args):
    root = Path(args.ptbxl_root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db = db[db.strat_fold.notna()].copy()
    if args.folds:
        folds = [int(x) for x in args.folds.split(",")]
        db = db[db.strat_fold.isin(folds)]
    if args.max_records:
        db = db.head(args.max_records)
    scp_map = load_scp_to_super(root / "scp_statements.csv")
    print(f"records: {len(db)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet1dWang(num_classes=5, input_channels=12).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # in-memory buffers (flush at end) — flat list of stage → list of (values, rec, bidx, tpos)
    buffers = {s: [] for s in ("stage1", "stage2", "stage3")}

    beat_meta_rows = []
    paths = [str(root / row.filename_lr) for _, row in db.iterrows()]
    ecg_ids = list(db.index)
    rows = list(db.itertuples())

    t0 = time.time()
    BATCH = args.batch

    # persistent threading pool avoids joblib fork-per-call overhead
    with Parallel(n_jobs=args.workers, backend="threading", prefer="threads") as pool:
        for bi in range(0, len(paths), BATCH):
            paths_b = paths[bi:bi + BATCH]
            loaded = pool(delayed(_load_record)(p) for p in paths_b)
            xs = np.stack([l[0] for l in loaded])

            with torch.no_grad():
                xb = torch.from_numpy(xs).to(device)
                h1 = model.stage1(model.stem(xb))
                h2 = model.stage2(h1)
                h3 = model.stage3(h2)
                stage_acts = {
                    "stage1": h1.float().cpu().numpy(),
                    "stage2": h2.float().cpu().numpy(),
                    "stage3": h3.float().cpu().numpy(),
                }

            for j in range(len(paths_b)):
                ecg_id = ecg_ids[bi + j]
                row = rows[bi + j]
                rpeaks = loaded[j][1]
                y = y_vec(row.scp_codes, scp_map)
                for bidx, r in enumerate(rpeaks):
                    beat_meta_rows.append({
                        "record_id": int(ecg_id), "beat_idx": int(bidx),
                        "fold": int(row.strat_fold),
                        "patient_id": int(row.patient_id),
                        **{c: float(y[k]) for k, c in enumerate(SUPERCLASSES)},
                        "r_sample_100hz": int(r),
                    })
                    for s, A in stage_acts.items():
                        T_s = A.shape[2]
                        r_s = int(round(int(r) / STAGE_DOWN[s]))
                        hw = STAGE_HALF[s]
                        lo, hi = r_s - hw, r_s + hw
                        if lo < 0 or hi > T_s:
                            continue
                        win = A[j, :, lo:hi].T.astype(np.float16)  # (2hw, 128)
                        n = win.shape[0]
                        buffers[s].append((
                            win,
                            np.full(n, int(ecg_id), dtype=np.int64),
                            np.full(n, int(bidx), dtype=np.int32),
                            np.arange(-hw, hw, dtype=np.int16),
                        ))

            if ((bi // BATCH) + 1) % 10 == 0:
                print(f"[{bi + len(paths_b)}/{len(paths)}] beats={len(beat_meta_rows)} "
                      f"elapsed={time.time()-t0:.1f}s "
                      f"rate={(bi + len(paths_b))/max(1,time.time()-t0):.1f} rec/s", flush=True)

    # flush HDF5 (no compression)
    for s in ("stage1", "stage2", "stage3"):
        parts = buffers[s]
        if not parts:
            continue
        vals = np.concatenate([p[0] for p in parts], axis=0)
        rec = np.concatenate([p[1] for p in parts])
        bidx = np.concatenate([p[2] for p in parts])
        tpos = np.concatenate([p[3] for p in parts])
        with h5py.File(out_dir / f"{s}_activations.h5", "w") as f:
            f.create_dataset("values", data=vals)
            f.create_dataset("record_id", data=rec)
            f.create_dataset("beat_idx", data=bidx)
            f.create_dataset("time_pos", data=tpos)
            f.attrs["fs_input"] = FS
            f.attrs["stage"] = s
            f.attrs["stage_downsample"] = STAGE_DOWN[s]
            f.attrs["stage_half_win"] = STAGE_HALF[s]
        print(f"[flush {s}] rows={vals.shape[0]:,}", flush=True)

    pd.DataFrame(beat_meta_rows).to_parquet(out_dir / "beat_meta.parquet", index=False)
    print(f"[done] beats={len(beat_meta_rows)} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--folds", default="")
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
