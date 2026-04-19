#!/usr/bin/env python
"""Fig 2: Top-activating ECG beats for selected SAE latents.

For each selected latent, find the 5 beats with the highest max-pooled
activation, load the original lead-II waveform, and overlay the per-timestep
SAE activation profile. Mark R-peak and PTB-XL superclass label.

Output: figures/fig2_top_activating_beats.pdf (6 panels × 5 beats grid)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from sae_topk import TopKSAE, topk_mask  # noqa: E402

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 100
STAGE_DOWN = {"stage1": 1, "stage2": 2, "stage3": 4}


def load_sae_and_encode_beats(sae_ckpt_path, acts_path, device="cpu"):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    sae = TopKSAE(ckpt["d_in"], ckpt["n_latents"], ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["model"])
    sae.eval()
    hook = ckpt["hook"]

    with h5py.File(acts_path, "r") as f:
        rec = f["record_id"][:]
        bidx = f["beat_idx"][:]
        n = rec.shape[0]
        composite = (rec.astype(np.int64) << 16) | (bidx.astype(np.int64) & 0xFFFF)
        uniq, assign = np.unique(composite, return_inverse=True)
        keys = np.stack([(uniq >> 16).astype(np.int64), (uniq & 0xFFFF).astype(np.int64)], axis=1)
        n_beats = len(keys)

        # per-beat max-pooled activation AND per-beat per-timestep activations for top beats
        pooled_max = np.zeros((n_beats, sae.n_latents), dtype=np.float32)
        BATCH = 200_000
        with torch.no_grad():
            for i in range(0, n, BATCH):
                end = min(i + BATCH, n)
                xb = torch.from_numpy(f["values"][i:end][:].astype(np.float32)).to(device)
                _, z, _ = sae(xb)
                z_np = z.cpu().numpy()
                idx = torch.from_numpy(assign[i:end].astype(np.int64)).to(device)
                idx_exp = idx.unsqueeze(1).expand(-1, sae.n_latents)
                pooled_t = torch.from_numpy(pooled_max).to(device)
                pooled_t.scatter_reduce_(0, idx_exp, z.cpu() if device == "cpu" else z, reduce="amax", include_self=True)
                pooled_max = pooled_t.numpy() if device == "cpu" else pooled_t.cpu().numpy()

    return sae, hook, pooled_max, keys, rec, bidx, assign


def get_beat_timeseries(sae, acts_path, target_rec, target_bidx, assign, rec_arr, bidx_arr, device="cpu"):
    """Get per-timestep SAE activation for a specific beat."""
    mask = (rec_arr == target_rec) & (bidx_arr == target_bidx)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    with h5py.File(acts_path, "r") as f:
        xb = torch.from_numpy(f["values"][indices[0]:indices[-1]+1][:].astype(np.float32)).to(device)
    with torch.no_grad():
        _, z, _ = sae(xb)
    return z.cpu().numpy()  # (T_beat, n_latents)


def load_ecg_window(ptbxl_root, ecg_id, r_sample, db):
    row = db.loc[ecg_id]
    sig, _ = wfdb.rdsamp(str(Path(ptbxl_root) / row.filename_lr))
    lead2 = sig[:, 1].astype(np.float32)
    # window +-0.5s around R-peak
    hw = 50  # 0.5s at 100Hz
    lo = max(0, r_sample - hw)
    hi = min(len(lead2), r_sample + hw)
    t = np.arange(lo, hi) / FS
    return t, lead2[lo:hi], lo, hi


def main(args):
    device = "cpu"
    ptbxl_root = Path(args.ptbxl_root)
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")
    meta = pd.read_parquet(args.beat_meta)

    sae, hook, pooled, keys, rec_arr, bidx_arr, assign = load_sae_and_encode_beats(
        args.sae_ckpt, args.acts, device)

    # Load interp summary to pick interesting latents
    summary = pd.read_parquet(args.latent_summary)
    mono = summary[summary.monosemantic_interplm]

    # Select 6 latents: diverse concepts
    selected = []
    # 1. Best left_axis latent
    la = mono[mono.best_concept == "left_axis"].sort_values("test_f1", ascending=False)
    if len(la): selected.append(("left_axis", int(la.iloc[0].latent)))
    # 2. Best NORM latent
    nm = mono[mono.best_concept == "NORM"].sort_values("test_f1", ascending=False)
    if len(nm): selected.append(("NORM", int(nm.iloc[0].latent)))
    # 3. Best MI latent (if any)
    mi = mono[mono.best_concept == "MI"].sort_values("test_f1", ascending=False)
    if len(mi): selected.append(("MI", int(mi.iloc[0].latent)))
    # 4. Best CD latent (if any)
    cd = mono[mono.best_concept == "CD"].sort_values("test_f1", ascending=False)
    if len(cd): selected.append(("CD", int(cd.iloc[0].latent)))
    # 5. Near-threshold wide_qrs (from binary interp)
    bin_df = pd.read_parquet(args.interp_binary)
    wq = bin_df[bin_df.concept == "wide_qrs"].sort_values("test_f1", ascending=False)
    if len(wq): selected.append(("wide_qrs (F1=%.2f)" % wq.iloc[0].test_f1, int(wq.iloc[0].latent)))
    # 6. Cross-class hub #1382 (or highest causal latent)
    selected.append(("hub #1382", 1382))

    # Pad to 6 if needed
    while len(selected) < 6 and len(nm) > len(selected):
        selected.append(("NORM-%d" % len(selected), int(nm.iloc[len(selected) - 1].latent)))

    print(f"Selected latents: {selected}", flush=True)

    N_TOP = 5
    fig = plt.figure(figsize=(16, 3.2 * len(selected)))
    gs = GridSpec(len(selected), N_TOP, figure=fig, hspace=0.4, wspace=0.3)

    for row_i, (concept_label, latent_j) in enumerate(selected):
        # top-5 beats by max activation of this latent
        acts_j = pooled[:, latent_j]
        top_idx = np.argsort(-acts_j)[:N_TOP]

        for col_i, bi in enumerate(top_idx):
            ecg_id = int(keys[bi, 0])
            beat_idx = int(keys[bi, 1])

            # find R-peak sample
            beat_row = meta[(meta.record_id == ecg_id) & (meta.beat_idx == beat_idx)]
            if len(beat_row) == 0:
                continue
            r_sample = int(beat_row.iloc[0].r_sample_100hz)
            superclass_labels = [c for c in SUPERCLASSES if beat_row.iloc[0].get(c, 0) > 0.5]

            t, ecg, lo, hi = load_ecg_window(ptbxl_root, ecg_id, r_sample, db)

            # per-timestep SAE activation for this beat
            z_ts = get_beat_timeseries(sae, args.acts, ecg_id, beat_idx,
                                       assign, rec_arr, bidx_arr, device)

            ax = fig.add_subplot(gs[row_i, col_i])
            ax.plot(t, ecg, "k-", lw=0.8, alpha=0.7)
            ax.axvline(r_sample / FS, color="red", lw=0.5, ls="--", alpha=0.5)

            if z_ts is not None:
                # map SAE timesteps to ECG time
                stage_down = STAGE_DOWN[hook]
                half_win = z_ts.shape[0] // 2
                t_sae = np.arange(-half_win, half_win) * stage_down / FS + r_sample / FS
                act_profile = z_ts[:, latent_j]
                ax2 = ax.twinx()
                ax2.fill_between(t_sae, 0, act_profile, alpha=0.3, color="orange")
                ax2.set_ylim(0, max(act_profile.max() * 1.3, 0.01))
                ax2.set_ylabel("act", fontsize=6, color="orange")
                ax2.tick_params(labelsize=5, colors="orange")

            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=5)
            if col_i == 0:
                ax.set_ylabel(f"latent {latent_j}\n{concept_label}", fontsize=7, fontweight="bold")
            if row_i == 0:
                ax.set_title(f"#{col_i+1}", fontsize=7)
            if row_i == len(selected) - 1:
                ax.set_xlabel("time (s)", fontsize=6)

            label_str = ",".join(superclass_labels) if superclass_labels else "?"
            ax.text(0.02, 0.95, f"id={ecg_id}\n{label_str}", transform=ax.transAxes,
                    fontsize=5, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle("Top-5 activating beats per selected SAE latent (stage2_r32_k32)",
                 fontsize=10, y=1.01)
    out = Path(args.fig_dir) / "fig2_top_activating_beats.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ckpt", required=True)
    p.add_argument("--acts", required=True)
    p.add_argument("--beat_meta", required=True)
    p.add_argument("--latent_summary", required=True)
    p.add_argument("--interp_binary", required=True)
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--fig_dir", default="figures")
    main(p.parse_args())
