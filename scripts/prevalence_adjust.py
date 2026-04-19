#!/usr/bin/env python
"""Post-processing: add prevalence-adjusted F1 columns to existing interp parquets.

For each concept, computes:
  - trivial_f1: 2*prev/(1+prev) (all-positive predictor F1)
  - adjusted_threshold: max(0.5, trivial_f1 + margin)
  - monosemantic_prevalence_adjusted: bool (val_f1 > 0.5 AND test_f1 > adjusted_threshold)

Does NOT require re-running SAE encoding or interp — works on existing parquet files.

Usage:
    python prevalence_adjust.py \
        --interp_binary results/interp/stage2_r32_k32_interp_binary.parquet \
        --clinical_gt results/clinical_gt_records.parquet \
        --beat_meta results/classifier/beat_meta.parquet \
        --margin 0.05
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(args):
    b = pd.read_parquet(args.interp_binary)
    gt = pd.read_parquet(args.clinical_gt).rename(columns={"ecg_id": "record_id"})
    meta = pd.read_parquet(args.beat_meta)
    merged = meta.merge(gt, on="record_id", how="left")
    test_mask = merged["fold"] == 10

    # Compute per-concept prevalence on test fold
    concepts = b.concept.unique()
    prev_map = {}
    for concept in concepts:
        if concept in merged.columns:
            y = merged.loc[test_mask, concept].to_numpy(dtype=float)
            valid = np.isfinite(y)
            prev = float(y[valid].mean()) if valid.sum() > 0 else 0
        else:
            prev = 0
        prev_map[concept] = prev

    # Add columns
    trivial_f1s = []
    adjusted_thresholds = []
    mono_adjusted = []
    for _, row in b.iterrows():
        prev = prev_map.get(row.concept, 0)
        trivial = 2 * prev / (1 + prev) if prev > 0 else 0
        adj_thr = max(0.5, trivial + args.margin)
        is_mono = (row.val_f1 > 0.5) and (row.test_f1 > adj_thr)
        trivial_f1s.append(trivial)
        adjusted_thresholds.append(adj_thr)
        mono_adjusted.append(is_mono)

    b["trivial_f1"] = trivial_f1s
    b["adjusted_threshold"] = adjusted_thresholds
    b["monosemantic_prevalence_adjusted"] = mono_adjusted

    out = Path(args.interp_binary).with_suffix(".prevalence_adjusted.parquet")
    b.to_parquet(out, index=False)
    print(f"saved to {out}", flush=True)

    # Summary
    for concept in sorted(concepts):
        sub = b[b.concept == concept]
        raw = int(((sub.val_f1 > 0.5) & (sub.test_f1 > 0.5)).sum())
        adj = int(sub.monosemantic_prevalence_adjusted.sum())
        prev = prev_map.get(concept, 0)
        triv = 2 * prev / (1 + prev) if prev > 0 else 0
        print(f"  {concept:25s}: prev={prev:.3f} trivF1={triv:.3f} "
              f"raw_mono={raw:4d} adj_mono={adj:4d}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interp_binary", required=True)
    p.add_argument("--clinical_gt", required=True)
    p.add_argument("--beat_meta", required=True)
    p.add_argument("--margin", type=float, default=0.05,
                   help="margin above trivial F1 for prevalence-adjusted threshold")
    main(p.parse_args())
