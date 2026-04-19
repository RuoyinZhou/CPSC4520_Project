#!/usr/bin/env python
"""Generate the 5 paper figures from SAE training + interpretation outputs.

Inputs:
    results/sae/*.json            (val_mse, var_x, r, k, hook per SAE run)
    results/interp/<tag>_interp_binary.parquet (F1 per latent x concept)
    results/interp/<tag>_latent_summary.parquet
    results/ablation/<tag>_ablation.parquet
    results/classifier/test_predictions.npz
    results/beat_meta.parquet
Outputs (in figures/):
    fig1_pareto_l0_vs_mse.pdf
    fig2_top_activating_beats.pdf
    fig3_feature_concept_f1_heatmap.pdf
    fig4_monosemantic_vs_expansion.pdf
    fig5_ablation_top10.pdf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_pareto(sae_dir: Path, out: Path):
    rows = []
    for jp in sae_dir.glob("*.json"):
        r = json.loads(jp.read_text())
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[pareto] no SAE json in {sae_dir}"); return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, hook in zip(axes, ["stage1", "stage2", "stage3"]):
        sub = df[df.hook == hook]
        if sub.empty: continue
        sc = ax.scatter(sub.k, sub.val_mse_over_var,
                        c=sub.r, s=50, cmap="viridis")
        ax.set_xlabel("k (TopK)"); ax.set_xscale("log", base=2)
        ax.set_title(hook)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("expansion r")
    axes[0].set_ylabel("val MSE / var(x)")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"[pareto] wrote {out}")


def fig_heatmap(interp_dir: Path, tag: str, out: Path):
    bin_df = pd.read_parquet(interp_dir / f"{tag}_interp_binary.parquet")
    piv = bin_df.pivot(index="latent", columns="concept", values="test_f1").fillna(0)
    # show only latents with at least one F1 > 0.3 to reduce clutter
    mask = piv.max(axis=1) > 0.3
    piv = piv[mask]
    if piv.empty:
        print(f"[heatmap] no latent > 0.3 for {tag}"); return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.15 * len(piv))))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks([]); ax.set_ylabel(f"SAE latent ({len(piv)} with max F1 > 0.3)")
    plt.colorbar(im, ax=ax, label="test F1")
    ax.set_title(f"Feature × concept F1 heatmap — {tag}")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"[heatmap] wrote {out}")


def fig_mono_vs_expansion(sae_dir: Path, interp_dir: Path, out: Path):
    rows = []
    for jp in sae_dir.glob("*.json"):
        r = json.loads(jp.read_text())
        tag = r["tag"]
        sp = interp_dir / f"{tag}_latent_summary.parquet"
        if not sp.exists(): continue
        s = pd.read_parquet(sp)
        rows.append({
            "hook": r["hook"], "r": r["r"], "k": r["k"],
            "n_latents": len(s),
            "frac_mono": float(s.monosemantic_interplm.mean()),
            "frac_mono_excl": float(s.monosemantic_with_margin.mean()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[mono_vs_exp] no interp outputs"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    for hook in sorted(df.hook.unique()):
        sub = df[df.hook == hook].groupby("r")[["frac_mono", "frac_mono_excl"]].mean().reset_index()
        ax.plot(sub.r, sub.frac_mono, "o-", label=f"{hook}")
    ax.set_xscale("log", base=2); ax.set_xlabel("expansion factor r")
    ax.set_ylabel("fraction monosemantic (InterPLM F1 > 0.5)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"[mono_vs_exp] wrote {out}")


def fig_ablation(abl_dir: Path, tag: str, out: Path):
    p = abl_dir / f"{tag}_ablation.parquet"
    if not p.exists():
        print(f"[ablation] missing {p}"); return
    df = pd.read_parquet(p)
    top = df.sort_values(["class", "delta_causal"], ascending=[True, False])\
            .groupby("class").head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = sorted(top["class"].unique())
    for i, c in enumerate(classes):
        sub = top[top["class"] == c].reset_index(drop=True)
        ax.bar(np.arange(len(sub)) + i * 11, sub.delta_causal, label=c)
    ax.set_xlabel("latent rank (within class)")
    ax.set_ylabel("Δ_causal = AUC_recon − AUC_ablated")
    ax.set_title(f"Top-10 causally important latents per class — {tag}")
    ax.legend(ncol=5, fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"[ablation] wrote {out}")


def main(a):
    fig_dir = Path(a.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig_pareto(Path(a.sae_dir), fig_dir / "fig1_pareto_l0_vs_mse.pdf")
    if a.best_tag:
        fig_heatmap(Path(a.interp_dir), a.best_tag, fig_dir / "fig3_feature_concept_f1_heatmap.pdf")
        fig_ablation(Path(a.ablation_dir), a.best_tag, fig_dir / "fig5_ablation_top10.pdf")
    fig_mono_vs_expansion(Path(a.sae_dir), Path(a.interp_dir),
                          fig_dir / "fig4_monosemantic_vs_expansion.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sae_dir", default="results/sae")
    p.add_argument("--interp_dir", default="results/interp")
    p.add_argument("--ablation_dir", default="results/ablation")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--best_tag", default="", help="tag of best SAE run (for heatmap + ablation figs)")
    main(p.parse_args())
