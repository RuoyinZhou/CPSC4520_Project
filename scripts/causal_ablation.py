#!/usr/bin/env python
"""Causal validation via SAE latent ablation.

For each shortlisted latent j we run three forward passes per fold-10 record:
  1) clean: classifier output p_clean
  2) recon: replace hook-site activation with its SAE round-trip, reproject
     through the rest of the classifier -> p_recon
  3) abl  : same as recon but with latent j zeroed -> p_abl

We report:
  * Delta_causal(j, c) = AUC_c(p_recon) - AUC_c(p_abl_j)
  * Delta_recon(c)    = AUC_c(p_clean) - AUC_c(p_recon)      (reconstruction tax)

Delta_causal isolates the latent's contribution net of reconstruction
imperfection. Per Anthropic 'Towards Monosemanticity' we report the
reconstruction baseline alongside the clean baseline.
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model_resnet1d_wang import ResNet1dWang  # noqa: E402
from sae_topk import TopKSAE, topk_mask  # noqa: E402

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_scp_to_super(scp_csv):
    df = pd.read_csv(scp_csv, index_col=0)
    df = df[df.diagnostic == 1.0]
    return {k: v for k, v in df.diagnostic_class.items() if v in SUPERCLASSES}


def y_vec(scp_codes, scp_map):
    if isinstance(scp_codes, str):
        scp_codes = ast.literal_eval(scp_codes)
    sc = {scp_map[c] for c in scp_codes if c in scp_map}
    return np.array([float(s in sc) for s in SUPERCLASSES], dtype=np.float32)


class PatchedForward:
    """Run resnet1d_wang with a user-supplied replacement at the hook stage's output."""

    def __init__(self, model: ResNet1dWang, hook: str):
        assert hook in ("stage1", "stage2", "stage3")
        self.model = model
        self.hook = hook

    def forward_with_replace(self, x: torch.Tensor, replace_fn):
        """replace_fn: callable(stage_out) -> new stage_out; runs on the hook output."""
        h = self.model.stem(x)
        h = self.model.stage1(h)
        if self.hook == "stage1":
            h = replace_fn(h)
        h = self.model.stage2(h)
        if self.hook == "stage2":
            h = replace_fn(h)
        h = self.model.stage3(h)
        if self.hook == "stage3":
            h = replace_fn(h)
        h = self.model.pool(h)
        return self.model.head(h)


def sae_replace(sae: TopKSAE, stage_out: torch.Tensor, zero_latent: int | None) -> torch.Tensor:
    """Pass (B, 128, T) through SAE acting per-timestep; optionally zero latent j."""
    B, C, T = stage_out.shape
    x = stage_out.permute(0, 2, 1).reshape(B * T, C)
    with torch.no_grad():
        pre = sae.encode_pre(x)
        z = topk_mask(pre, sae.k)
        if zero_latent is not None:
            z[:, zero_latent] = 0.0
        recon = sae.decoder(z) + sae.pre_bias
    return recon.reshape(B, T, C).permute(0, 2, 1).contiguous()


def sae_replace_multi(sae: TopKSAE, stage_out: torch.Tensor, zero_latents: list[int]) -> torch.Tensor:
    """Like sae_replace but zeros multiple latents at once (grouped ablation)."""
    B, C, T = stage_out.shape
    x = stage_out.permute(0, 2, 1).reshape(B * T, C)
    with torch.no_grad():
        pre = sae.encode_pre(x)
        z = topk_mask(pre, sae.k)
        for j in zero_latents:
            z[:, j] = 0.0
        recon = sae.decoder(z) + sae.pre_bias
    return recon.reshape(B, T, C).permute(0, 2, 1).contiguous()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load model + SAE
    ckpt = torch.load(args.classifier_ckpt, map_location=device)
    model = ResNet1dWang(num_classes=5, input_channels=12).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sae_ckpt = torch.load(args.sae_ckpt, map_location=device)
    sae = TopKSAE(sae_ckpt["d_in"], sae_ckpt["n_latents"], sae_ckpt["k"]).to(device)
    sae.load_state_dict(sae_ckpt["model"])
    sae.eval()

    hook = sae_ckpt["hook"]
    patch = PatchedForward(model, hook)

    # test set (fold 10)
    root = Path(args.ptbxl_root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db = db[db.strat_fold == 10].copy()
    scp_map = load_scp_to_super(root / "scp_statements.csv")

    # shortlist latents: top-N per concept from feature_interp output
    shortlist_df = pd.read_json(args.shortlist)
    latents = sorted(set(int(j) for j in shortlist_df["latent"]))
    print(f"hook={hook}  latents-to-ablate={len(latents)}  test-records={len(db)}", flush=True)

    # collect signals once
    X_list, Y_list, ids = [], [], []
    for ecg_id, row in db.iterrows():
        sig, _ = wfdb.rdsamp(str(root / row.filename_lr))
        x = sig.astype(np.float32).T
        if x.shape[1] != 1000:
            pad = 1000 - x.shape[1]
            x = np.pad(x, ((0, 0), (0, max(0, pad))))[:, :1000]
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        X_list.append(x); Y_list.append(y_vec(row.scp_codes, scp_map)); ids.append(int(ecg_id))
    X = torch.from_numpy(np.stack(X_list)).to(device)
    Y = np.stack(Y_list)
    print(f"test tensor: {X.shape}", flush=True)

    # baseline: clean
    def probs_for(fn):
        outs = []
        for i in range(0, len(X), args.batch_size):
            xb = X[i:i+args.batch_size]
            with torch.no_grad():
                logits = fn(xb)
            outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs)

    p_clean = probs_for(lambda xb: model(xb))
    p_recon = probs_for(lambda xb: patch.forward_with_replace(xb,
        lambda act: sae_replace(sae, act, None)))
    auc_clean = {c: float(roc_auc_score(Y[:, i], p_clean[:, i]))
                 if Y[:, i].sum() else float("nan")
                 for i, c in enumerate(SUPERCLASSES)}
    auc_recon = {c: float(roc_auc_score(Y[:, i], p_recon[:, i]))
                 if Y[:, i].sum() else float("nan")
                 for i, c in enumerate(SUPERCLASSES)}
    delta_recon = {c: auc_clean[c] - auc_recon[c] for c in SUPERCLASSES}
    print("delta_recon:", delta_recon, flush=True)

    # helper: AUC on a subset of samples
    def auc_subset(y_col, p_col, mask):
        if mask.sum() < 2 or y_col[mask].sum() == 0 or y_col[mask].sum() == mask.sum():
            return float("nan")
        return float(roc_auc_score(y_col[mask], p_col[mask]))

    # helper: bootstrap CI on delta
    def bootstrap_ci(y_col, p1, p2, n_boot=1000, ci=0.95):
        rng = np.random.RandomState(42)
        n = len(y_col)
        deltas = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yb, p1b, p2b = y_col[idx], p1[idx], p2[idx]
            if yb.sum() == 0 or yb.sum() == len(yb):
                deltas[b] = 0.0
                continue
            deltas[b] = roc_auc_score(yb, p1b) - roc_auc_score(yb, p2b)
        lo = np.percentile(deltas, (1 - ci) / 2 * 100)
        hi = np.percentile(deltas, (1 + ci) / 2 * 100)
        return float(np.mean(deltas)), float(lo), float(hi)

    # per-latent ablation (single-latent)
    rows = []
    per_sample_deltas = {}  # latent -> (n_samples, 5) array of per-sample prob diffs
    t0 = time.time()
    for n, j in enumerate(latents):
        p_abl = probs_for(lambda xb, jj=j: patch.forward_with_replace(xb,
            lambda act: sae_replace(sae, act, jj)))
        per_sample_deltas[j] = p_recon - p_abl  # (N, 5) per-sample Δ
        for i, c in enumerate(SUPERCLASSES):
            if Y[:, i].sum() == 0:
                continue
            auc_abl = float(roc_auc_score(Y[:, i], p_abl[:, i]))
            # positive-only AUC (commentor Action 2A)
            pos_mask = Y[:, i] == 1
            neg_mask = Y[:, i] == 0
            delta_pos = float(np.mean((p_recon[:, i] - p_abl[:, i])[pos_mask]))
            delta_neg = float(np.mean((p_recon[:, i] - p_abl[:, i])[neg_mask]))
            # bootstrap CI on delta_causal
            mean_boot, ci_lo, ci_hi = bootstrap_ci(Y[:, i], p_recon[:, i], p_abl[:, i])
            rows.append({
                "hook": hook, "latent": j, "class": c,
                "auc_clean": auc_clean[c],
                "auc_recon": auc_recon[c],
                "auc_abl": auc_abl,
                "delta_causal": auc_recon[c] - auc_abl,
                "delta_causal_pos_mean": delta_pos,
                "delta_causal_neg_mean": delta_neg,
                "delta_causal_boot_mean": mean_boot,
                "delta_causal_ci95_lo": ci_lo,
                "delta_causal_ci95_hi": ci_hi,
                "delta_recon": delta_recon[c],
                "n_pos": int(pos_mask.sum()),
                "n_neg": int(neg_mask.sum()),
            })
        if (n + 1) % 10 == 0:
            print(f"[{n+1}/{len(latents)}] elapsed={time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"{args.tag}_ablation.parquet", index=False)

    # headline: top-10 causally important per class
    head = (df.sort_values(["class", "delta_causal"], ascending=[True, False])
              .groupby("class").head(10))
    head.to_csv(out_dir / f"{args.tag}_top10_per_class.csv", index=False)
    print(head.to_string(index=False), flush=True)

    # save per-sample Δ for top latents (for histogram plots)
    top_latents_per_class = {}
    for c in SUPERCLASSES:
        sub = df[df["class"] == c].nlargest(5, "delta_causal")
        top_latents_per_class[c] = sub["latent"].tolist()
    per_sample_out = {}
    for c, lat_list in top_latents_per_class.items():
        ci = SUPERCLASSES.index(c)
        for j in lat_list:
            if j in per_sample_deltas:
                key = f"lat{j}_{c}"
                per_sample_out[key] = per_sample_deltas[j][:, ci]
    np.savez(out_dir / f"{args.tag}_per_sample_delta.npz",
             y_true=Y, classes=np.array(SUPERCLASSES), **per_sample_out)
    print(f"saved per-sample deltas for {len(per_sample_out)} latent-class pairs", flush=True)

    # grouped ablation: ablate top-K latents together (K=5, 10, 20)
    print("\n=== Grouped ablation ===", flush=True)
    grouped_rows = []
    for c in SUPERCLASSES:
        ci = SUPERCLASSES.index(c)
        if Y[:, ci].sum() == 0:
            continue
        # rank latents by single-latent delta_causal for this class
        sub = df[df["class"] == c].sort_values("delta_causal", ascending=False)
        ranked_latents = sub["latent"].tolist()
        for K in [5, 10, 20]:
            top_k = ranked_latents[:min(K, len(ranked_latents))]
            p_abl_grp = probs_for(lambda xb, jj=top_k: patch.forward_with_replace(xb,
                lambda act, jjj=jj: sae_replace_multi(sae, act, jjj)))
            auc_grp = float(roc_auc_score(Y[:, ci], p_abl_grp[:, ci]))
            delta_grp = auc_recon[c] - auc_grp
            # bootstrap CI
            mean_boot, ci_lo, ci_hi = bootstrap_ci(Y[:, ci], p_recon[:, ci], p_abl_grp[:, ci])
            grouped_rows.append({
                "hook": hook, "class": c, "K": K,
                "n_latents_ablated": len(top_k),
                "auc_recon": auc_recon[c], "auc_abl_grouped": auc_grp,
                "delta_causal_grouped": delta_grp,
                "delta_grouped_boot_mean": mean_boot,
                "delta_grouped_ci95_lo": ci_lo,
                "delta_grouped_ci95_hi": ci_hi,
                "latents": top_k,
            })
            print(f"  [{c}] top-{K}: Δ_grouped={delta_grp:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]", flush=True)
    grouped_df = pd.DataFrame(grouped_rows)
    grouped_df.to_parquet(out_dir / f"{args.tag}_grouped_ablation.parquet", index=False)

    # random-latent ablation null baseline: ablate k random latents (50 draws)
    # to establish the null distribution for grouped ablation
    print("\n=== Random-latent ablation null baseline ===", flush=True)
    rng = np.random.RandomState(42)
    n_draws = 50
    all_latent_ids = list(range(sae.n_latents))
    null_rows = []
    for K in [5, 10, 20]:
        for c in SUPERCLASSES:
            ci = SUPERCLASSES.index(c)
            if Y[:, ci].sum() == 0:
                continue
            deltas = np.zeros(n_draws)
            for d in range(n_draws):
                rand_latents = sorted(rng.choice(all_latent_ids, size=K, replace=False).tolist())
                p_abl_rand = probs_for(lambda xb, jj=rand_latents: patch.forward_with_replace(xb,
                    lambda act, jjj=jj: sae_replace_multi(sae, act, jjj)))
                auc_rand = float(roc_auc_score(Y[:, ci], p_abl_rand[:, ci]))
                deltas[d] = auc_recon[c] - auc_rand
            null_rows.append({
                "hook": hook, "class": c, "K": K,
                "null_mean": float(deltas.mean()),
                "null_std": float(deltas.std()),
                "null_p95": float(np.percentile(deltas, 95)),
                "null_p99": float(np.percentile(deltas, 99)),
                "top_k_delta": float(grouped_df[(grouped_df["class"] == c) & (grouped_df.K == K)].delta_causal_grouped.iloc[0])
                    if len(grouped_df[(grouped_df["class"] == c) & (grouped_df.K == K)]) > 0 else np.nan,
            })
            print(f"  [{c}] K={K}: null_mean={deltas.mean():.4f}±{deltas.std():.4f} "
                  f"p95={np.percentile(deltas, 95):.4f} "
                  f"top-K={null_rows[-1]['top_k_delta']:.4f}", flush=True)
    null_df = pd.DataFrame(null_rows)
    null_df["exceeds_p95"] = null_df.top_k_delta > null_df.null_p95
    null_df["exceeds_p99"] = null_df.top_k_delta > null_df.null_p99
    null_df.to_parquet(out_dir / f"{args.tag}_random_null.parquet", index=False)
    null_df.to_csv(out_dir / f"{args.tag}_random_null.csv", index=False)
    print(null_df.to_string(index=False), flush=True)

    print(f"\n[{args.tag}] DONE — saved ablation + grouped + per-sample + random null", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--classifier_ckpt", required=True)
    p.add_argument("--sae_ckpt", required=True)
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--shortlist", required=True, help="JSON list of {latent, concept, ...} to ablate")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
