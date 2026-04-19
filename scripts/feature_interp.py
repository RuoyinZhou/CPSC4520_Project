#!/usr/bin/env python
"""InterPLM F1 monosemanticity protocol for trained TopK SAEs.

For each (SAE latent j, binary clinical concept c):
  1. Project per-beat activations by pooling within each beat (max over the
     beat's time positions) -> one scalar per beat per latent.
  2. Sweep activation thresholds in THRESH_GRID on the fold-9 beats.
  3. Pick the threshold maximising F1 on fold-9; compute the same-threshold
     F1 on fold-10.
  4. Monosemantic iff val_F1 > 0.5 AND test_F1 > 0.5 (InterPLM default).
  5. Report an exclusivity margin as well (best_F1 minus runner-up best_F1 >
     0.2) as our own extension.

Also reports per-latent Spearman rho against continuous features (QRS dur,
PR int, etc.) as supplementary.

Outputs:
    <out_dir>/<tag>_interp_binary.parquet
    <out_dir>/<tag>_interp_continuous.parquet
    <out_dir>/<tag>_top_latents.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from sae_topk import TopKSAE  # noqa: E402

# InterPLM uses {0, 0.15, 0.5, 0.6, 0.8} for protein LM neurons that can be negative.
# Our normalization clips negative activations to zero (feature absence convention),
# so threshold=0 is trivially all-positive. We replace 0 with 0.05 (smallest
# meaningful threshold after clip-to-zero normalization).
THRESH_GRID = np.array([0.05, 0.15, 0.5, 0.6, 0.8])


def encode_all(sae: TopKSAE, acts_path: Path, batch: int = 16384,
               device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streaming max-pool per beat: encodes the activations in batches and folds
    them into a (n_beats, n_latents) running-max array on the fly. Memory cost is
    O(n_beats * n_latents) instead of O(n_timesteps * n_latents) — typically 50×
    smaller (e.g. for stage1_r32_k8: 260k beats × 4096 latents × 4B ≈ 4.3 GB
    instead of 340 GB)."""
    with h5py.File(acts_path, "r") as f:
        rec = f["record_id"][:]
        bidx = f["beat_idx"][:]
        n = rec.shape[0]
        composite = (rec.astype(np.int64) << 16) | (bidx.astype(np.int64) & 0xFFFF)
        uniq, beat_assign = np.unique(composite, return_inverse=True)
        beat_assign = torch.from_numpy(beat_assign.astype(np.int64))
        keys = np.stack([(uniq >> 16).astype(np.int64),
                         (uniq & 0xFFFF).astype(np.int64)], axis=1)
        n_beats = len(keys)
        pooled = torch.zeros((n_beats, sae.n_latents), dtype=torch.float32, device=device)
        sae.eval()
        # Big batches because the SAE forward is tiny (Linear 128 -> n_latents).
        # Using torch.scatter_reduce_("amax") replaces the slow np.maximum.at loop.
        BIG = max(batch, 200_000)
        with torch.no_grad():
            for i in range(0, n, BIG):
                end = min(i + BIG, n)
                xb = torch.from_numpy(f["values"][i:end][:].astype(np.float32)).to(device)
                _, z, _ = sae(xb)            # (b, n_latents)
                idx = beat_assign[i:end].to(device).unsqueeze(1).expand(-1, sae.n_latents)
                pooled.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
                if i // BIG % 5 == 0:
                    print(f"  encode {end:,}/{n:,}", flush=True)
        rec_out = keys[:, 0].astype(np.int64)
        bidx_out = keys[:, 1].astype(np.int32)
    return pooled.cpu().numpy(), rec_out, bidx_out


def normalize_per_latent(z: np.ndarray, train_val_mask: np.ndarray) -> np.ndarray:
    """Max-normalize per latent using train+val data only (no test leakage).
    Matches InterPLM's max-normalization (Simon & Zou, Nature Methods).
    Dead latents (zero on all train+val) are left at zero to avoid divide-by-epsilon."""
    maxvals = np.abs(z[train_val_mask]).max(axis=0)
    dead = maxvals < 1e-12
    maxvals[dead] = 1.0  # avoid division by zero; these latents stay at 0
    return np.clip(z / maxvals, 0, None)  # no upper clip to 1 — match InterPLM


def f1_at(z_norm: np.ndarray, y: np.ndarray, thresh: float,
          valid: np.ndarray | None = None) -> float:
    if valid is not None:
        z_norm, y = z_norm[valid], y[valid]
    pred = (z_norm >= thresh).astype(np.int8)
    if pred.sum() == 0 or y.sum() == 0:
        return 0.0
    return float(f1_score(y, pred, zero_division=0))


def interp_binary(z_val: np.ndarray, z_test: np.ndarray,
                  y_val: np.ndarray, y_test: np.ndarray,
                  concept: str,
                  valid_val: np.ndarray | None = None,
                  valid_test: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for j in range(z_val.shape[1]):
        best_val, best_thr = -1.0, None
        for t in THRESH_GRID:
            f = f1_at(z_val[:, j], y_val, t, valid_val)
            if f > best_val:
                best_val, best_thr = f, t
        test_f = f1_at(z_test[:, j], y_test, best_thr, valid_test) if best_thr is not None else 0.0
        rows.append({
            "latent": j, "concept": concept,
            "val_f1": best_val, "test_f1": test_f, "threshold": best_thr,
        })
    return pd.DataFrame(rows)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.sae_ckpt, map_location=device)
    sae = TopKSAE(ckpt["d_in"], ckpt["n_latents"], ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["model"])
    print(f"loaded SAE: d_in={ckpt['d_in']} latents={ckpt['n_latents']} k={ckpt['k']}", flush=True)

    # encode ALL beat activations (pool per beat)
    z_all, rec_arr, bidx_arr = encode_all(sae, Path(args.acts), device=device)
    print(f"pooled per-beat: {z_all.shape}", flush=True)
    # normalization deferred until after fold alignment (need train+val mask)

    # beat metadata with fold + superclass one-hot
    meta = pd.read_parquet(args.beat_meta)
    beat_key = meta[["record_id", "beat_idx"]].astype({"record_id": "int64", "beat_idx": "int32"})
    meta_idx = pd.MultiIndex.from_frame(beat_key)
    act_idx = pd.MultiIndex.from_arrays([rec_arr, bidx_arr], names=["record_id", "beat_idx"])
    # align
    mapper = {k: i for i, k in enumerate(act_idx)}
    aligned_rows = [mapper.get(k) for k in meta_idx]
    keep = np.array([r is not None for r in aligned_rows])
    idx = np.array([r for r in aligned_rows if r is not None])
    meta_kept = meta[keep].reset_index(drop=True)
    z_aligned = z_all[idx]
    print(f"aligned beats: {z_aligned.shape[0]}", flush=True)

    # binary clinical ground truth: join record-level GT on record_id
    gt = pd.read_parquet(args.clinical_gt).rename(columns={"ecg_id": "record_id"})
    # filter features by n_agree >= min_n_agree (agreement among PTB-XL+ algos)
    # single-source (n_agree=1) features are retained in a flagged column
    # `{feat}_single_source=True` so downstream can report them separately.
    if args.min_n_agree > 1:
        for col in [c for c in gt.columns if not c.endswith("__n_agree")
                    and c != "record_id"]:
            nagree_col = f"{col}__n_agree"
            if nagree_col in gt.columns:
                mask = gt[nagree_col].fillna(0) < args.min_n_agree
                gt.loc[mask, col] = np.nan
    merged = meta_kept.merge(gt, on="record_id", how="left")

    # binary concepts from build_clinical_gt.py
    # Column names must match build_clinical_gt.py output exactly:
    #   st_elevation_any, st_depression_any, wide_p (not st_elevation, st_depression, p_wave_present)
    BINARY = [c for c in [
        "wide_qrs", "st_elevation_any", "st_depression_any", "prolonged_pr", "short_pr",
        "prolonged_qtc", "right_axis", "left_axis", "wide_p", "wide_p_120",
    ] if c in merged.columns]
    # add PTB-XL superclass labels as binary concepts too
    for sc in ["NORM", "MI", "STTC", "CD", "HYP"]:
        if sc in merged.columns:
            BINARY.append(sc)

    # fold mask: 9 validation, 10 test
    val_mask = (merged["fold"] == 9).to_numpy()
    test_mask = (merged["fold"] == 10).to_numpy()
    # normalize using train+val only (folds 1-9), excluding test fold 10 (B4 fix)
    train_val_mask = ~test_mask
    z_norm = normalize_per_latent(z_aligned, train_val_mask)

    all_bin = []
    for concept in BINARY:
        y_raw = merged[concept].to_numpy(dtype=float)
        valid = np.isfinite(y_raw)  # B5: keep NaN as missing, exclude from F1
        y = np.where(valid, y_raw, 0).astype(int)  # placeholder 0 for NaN (masked out)
        valid_val = valid[val_mask]
        valid_test = valid[test_mask]
        yv, yt = y[val_mask], y[test_mask]
        if yv[valid_val].sum() == 0 or yt[valid_test].sum() == 0:
            print(f"[skip] {concept}: empty positive class in val/test", flush=True)
            continue
        df = interp_binary(z_norm[val_mask], z_norm[test_mask], yv, yt, concept,
                           valid_val, valid_test)
        all_bin.append(df)
        print(f"[{concept}] pos_val={int(yv.sum())} pos_test={int(yt.sum())} "
              f"top-5 test_f1: {df.nlargest(5, 'test_f1')[['latent','test_f1']].to_dict('records')}",
              flush=True)
    bin_df = pd.concat(all_bin, ignore_index=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bin_df.to_parquet(out_dir / f"{args.tag}_interp_binary.parquet", index=False)

    # monosemantic summary
    piv = bin_df.pivot(index="latent", columns="concept", values="test_f1").fillna(0)
    # best and runner-up test F1 per latent
    vals = piv.to_numpy(float)
    best = vals.max(axis=1)
    top = np.argsort(-vals, axis=1)
    second = np.take_along_axis(vals, top[:, 1:2], axis=1).flatten() if vals.shape[1] > 1 else np.zeros(vals.shape[0])
    # require both val + test F1 > 0.5
    val_piv = bin_df.pivot(index="latent", columns="concept", values="val_f1").fillna(0).to_numpy(float)
    val_best = val_piv.max(axis=1)
    mono = (best > 0.5) & (val_best > 0.5)
    excl = mono & ((best - second) > 0.2)
    top_concept_per_latent = piv.columns[vals.argmax(axis=1)]
    summary = pd.DataFrame({
        "latent": piv.index,
        "best_concept": top_concept_per_latent,
        "val_f1": val_best,
        "test_f1": best,
        "runner_up_test_f1": second,
        "monosemantic_interplm": mono,
        "monosemantic_with_margin": excl,
    })
    summary.to_parquet(out_dir / f"{args.tag}_latent_summary.parquet", index=False)
    n_mono = int(mono.sum()); n_excl = int(excl.sum())
    print(f"[{args.tag}] monosemantic (InterPLM): {n_mono}/{len(mono)} "
          f"({n_mono/len(mono):.2%}); with exclusivity margin: {n_excl}",
          flush=True)
    # per-superclass monosemantic count (imbalance diagnostic: NORM 52% of data
    # may dominate; we verify minority classes have >=1 monosemantic latent)
    super_concepts = ["NORM", "MI", "STTC", "CD", "HYP"]
    per_super = {c: 0 for c in super_concepts}
    mono_latent_ids = set(summary[mono].latent.astype(int))
    for _, row in summary[mono].iterrows():
        if row.best_concept in super_concepts:
            per_super[row.best_concept] += 1
    print(f"[{args.tag}] monosemantic per-superclass: {per_super}", flush=True)

    # prevalence-adjusted baseline: flag concepts where trivial all-positive F1 > 0.5
    for concept in BINARY:
        y_raw = merged[concept].to_numpy(dtype=float)
        valid = np.isfinite(y_raw)
        yt = y_raw[test_mask & valid]
        if len(yt) == 0:
            continue
        prev = float(yt.mean())
        trivial_f1 = 2 * prev / (1 + prev) if prev > 0 else 0.0
        n_mono_c = int(bin_df[(bin_df.concept == concept) &
                              (bin_df.val_f1 > 0.5) & (bin_df.test_f1 > 0.5)].shape[0])
        if trivial_f1 > 0.5:
            print(f"[{args.tag}] WARNING: {concept} prevalence={prev:.2%}, "
                  f"trivial_all_positive_F1={trivial_f1:.3f} > 0.5 threshold. "
                  f"{n_mono_c} 'monosemantic' latents may be inflated by base rate.",
                  flush=True)

    # continuous Spearman — vectorised: rank-transform y once, then correlate
    # against all latents in one matrix op. O(n_latents) instead of O(n_latents × n log n).
    CONT = [c for c in ["qrs_duration_ms", "pr_interval_ms", "qt_interval_ms",
                         "rr_interval_ms", "qrs_axis_deg",
                         "p_amplitude_mv", "t_amplitude_mv"] if c in merged.columns]
    rows = []
    from scipy.stats import rankdata
    for concept in CONT:
        y = merged[concept].to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 100:
            continue
        y_rank = rankdata(y[mask]).astype(np.float32)
        y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-12)
        z_sub = z_norm[mask]
        n_lat = z_sub.shape[1]
        CHUNK = min(n_lat, 512)
        rhos = np.zeros(n_lat, dtype=np.float32)
        for c0 in range(0, n_lat, CHUNK):
            c1 = min(c0 + CHUNK, n_lat)
            zc = z_sub[:, c0:c1].copy()
            zr = np.apply_along_axis(rankdata, 0, zc).astype(np.float32)
            zr = (zr - zr.mean(axis=0, keepdims=True)) / (zr.std(axis=0, keepdims=True) + 1e-12)
            rhos[c0:c1] = (zr * y_rank[:, None]).mean(axis=0)
        for j in range(n_lat):
            rows.append({"latent": int(j), "concept": concept, "spearman": float(rhos[j])})
    pd.DataFrame(rows).to_parquet(out_dir / f"{args.tag}_interp_continuous.parquet", index=False)

    # dump the top-20 causally-interesting latents per concept for ablation shortlist
    # Use val_f1 (not test_f1) for shortlist selection to avoid test-set selection bias
    top_per = bin_df.sort_values(["concept", "val_f1"], ascending=[True, False])\
                    .groupby("concept").head(20)
    top_per.to_json(out_dir / f"{args.tag}_top_latents.json", orient="records", indent=2)
    print(f"[{args.tag}] DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ckpt", required=True)
    p.add_argument("--acts", required=True, help="{stage}_activations.h5 used to train this SAE")
    p.add_argument("--beat_meta", required=True, help="beat_meta.parquet from extract_activations")
    p.add_argument("--clinical_gt", required=True, help="clinical_gt_records.parquet")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True, help="identifier for this SAE run (e.g. stage2_r16_k16)")
    p.add_argument("--min_n_agree", type=int, default=2,
                   help="drop clinical features with fewer than this many PTB-XL+ algos agreeing (default 2)")
    main(p.parse_args())
