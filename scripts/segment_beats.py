#!/usr/bin/env python
"""Segment PTB-XL records into per-beat windows.

Loads 500 Hz PTB-XL records, detects R-peaks via neurokit2, extracts fixed-length
beat windows (+-0.4 s = 400 samples) from lead II, filters by ecg_quality >= 0.5,
and writes a per-fold HDF5 containing beats, record_id, beat_idx, fold, patient_id,
and the full multi-label superclass one-hot for downstream classifier + SAE use.

Outputs (per fold in data/beats/):
    beats_lead2_fold{1..10}.h5   with datasets:
        beats (N, 400) float32
        record_id (N,) int64
        beat_idx (N,) int32
        patient_id (N,) int64
        superclass (N, 5) float32  (NORM, MI, STTC, CD, HYP)
"""
from __future__ import annotations
import argparse, ast, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import h5py

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 500
HALF_WIN_S = 0.4
HALF_WIN = int(FS * HALF_WIN_S)  # 200 samples
WIN = 2 * HALF_WIN               # 400 samples
QUALITY_THRESH = 0.5


def load_scp_to_super(scp_csv: Path) -> dict[str, str]:
    df = pd.read_csv(scp_csv, index_col=0)
    df = df[df.diagnostic == 1.0]
    return {k: v for k, v in df.diagnostic_class.items() if v in SUPERCLASSES}


def record_superclass_vec(scp_codes: dict, mapping: dict[str, str]) -> np.ndarray:
    sc = {mapping[c] for c in scp_codes if c in mapping}
    return np.array([float(s in sc) for s in SUPERCLASSES], dtype=np.float32)


def segment_record(sig: np.ndarray, fs: int) -> np.ndarray:
    """Return (N_beats,) int array of R-peak sample indices."""
    cleaned = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    rpeaks = info.get("ECG_R_Peaks", np.array([], dtype=int))
    return np.asarray(rpeaks, dtype=int)


def segment_one(record_path: Path, lead_idx: int = 1):
    rec = wfdb.rdrecord(str(record_path))
    sig = rec.p_signal[:, lead_idx].astype(np.float32)
    fs = int(rec.fs)
    rpeaks = segment_record(sig, fs)
    beats, keep_idx, qualities = [], [], []
    for i, r in enumerate(rpeaks):
        if r - HALF_WIN < 0 or r + HALF_WIN > len(sig):
            continue
        w = sig[r - HALF_WIN : r + HALF_WIN]
        q = float(nk.ecg_quality(w, sampling_rate=fs, method="averageQRS").mean())
        if q < QUALITY_THRESH or not np.isfinite(q):
            continue
        beats.append(w)
        keep_idx.append(i)
        qualities.append(q)
    if not beats:
        return np.empty((0, WIN), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
    return (
        np.stack(beats).astype(np.float32),
        np.asarray(keep_idx, dtype=np.int32),
        np.asarray(qualities, dtype=np.float32),
    )


def main(args):
    root = Path(args.ptbxl_root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp_map = load_scp_to_super(root / "scp_statements.csv")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(1, 11))
    t0 = time.time()
    for fold in selected_folds:
        sub = db[db.strat_fold == fold]
        print(f"[fold {fold}] {len(sub)} records", flush=True)
        all_beats, all_rec, all_bidx, all_pid, all_y = [], [], [], [], []
        for i, (ecg_id, row) in enumerate(sub.iterrows()):
            rec_path = root / row.filename_hr  # 500 Hz
            try:
                beats, bidx, _ = segment_one(rec_path, lead_idx=args.lead)
            except Exception as e:
                print(f"  !! {ecg_id}: {e}", file=sys.stderr, flush=True)
                continue
            if beats.size == 0:
                continue
            y = record_superclass_vec(row.scp_codes, scp_map)
            n = beats.shape[0]
            all_beats.append(beats)
            all_rec.append(np.full(n, ecg_id, dtype=np.int64))
            all_bidx.append(bidx.astype(np.int32))
            all_pid.append(np.full(n, int(row.patient_id), dtype=np.int64))
            all_y.append(np.tile(y, (n, 1)))
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(sub)}] beats so far={sum(b.shape[0] for b in all_beats)} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        if not all_beats:
            print(f"[fold {fold}] EMPTY — skipping write", flush=True)
            continue
        beats = np.concatenate(all_beats); rec = np.concatenate(all_rec)
        bidx = np.concatenate(all_bidx); pid = np.concatenate(all_pid)
        y = np.concatenate(all_y)
        out = out_dir / f"beats_lead{args.lead}_fold{fold}.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("beats", data=beats, compression="gzip", compression_opts=4)
            f.create_dataset("record_id", data=rec)
            f.create_dataset("beat_idx", data=bidx)
            f.create_dataset("patient_id", data=pid)
            f.create_dataset("superclass", data=y)
            f.attrs["fs"] = FS
            f.attrs["win_samples"] = WIN
            f.attrs["lead_idx"] = args.lead
            f.attrs["superclasses"] = np.array(SUPERCLASSES, dtype="S")
            f.attrs["quality_thresh"] = QUALITY_THRESH
        print(f"[fold {fold}] wrote {out}: {beats.shape}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lead", type=int, default=1, help="0-indexed lead (1 = lead II)")
    p.add_argument("--folds", default="", help="comma-separated subset of folds 1..10 (default all)")
    main(p.parse_args())
