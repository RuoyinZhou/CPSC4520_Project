#!/usr/bin/env python
"""TopK sparse autoencoder with AuxK dead-latent revival.

Implements the Gao et al. 2024 (arXiv:2406.04093) TopK SAE:
    z = TopK_k(W_enc(x - b_pre))
    x_hat = W_dec z + b_pre
with:
    - decoder columns renormalized to unit norm after every optimizer step
    - AuxK loss: L_aux = || (x - x_hat) - W_dec z_aux ||^2, with z_aux =
      TopK_{k_aux} restricted to currently-DEAD latents, coefficient alpha=1/32
    - dead latent = no nonzero activation for the last N samples (default 1
      full epoch, our scaled-down analogue of Gao's 10M-token window)

CLI trains one SAE at a given (hook, r, k) and writes the checkpoint plus a
per-step metrics CSV to enable Pareto selection.
"""
from __future__ import annotations
import argparse, os, time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


def topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    # keep top-k per row, zero rest
    if k >= z.shape[-1]:
        return z
    vals, idx = torch.topk(z, k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, vals)
    return out


def topk_on_restricted(z: torch.Tensor, k: int, allow_mask: torch.Tensor) -> torch.Tensor:
    """TopK_k over only latents where allow_mask is True (1D, shape [n_latents]).
    Used for AuxK to restrict to currently-dead latents."""
    masked = z.masked_fill(~allow_mask[None, :], float("-inf"))
    eff_k = int(min(k, int(allow_mask.sum().item())))
    if eff_k <= 0:
        return torch.zeros_like(z)
    vals, idx = torch.topk(masked, eff_k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, vals)
    # replace -inf survivors (should not remain, but be safe)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, k: int):
        super().__init__()
        self.d_in = d_in; self.n_latents = n_latents; self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)
        # init: kaiming for encoder; decoder columns unit-norm
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            W = torch.randn(d_in, n_latents)
            W /= W.norm(dim=0, keepdim=True) + 1e-8
            self.decoder.weight.copy_(W)

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight  # (d_in, n_latents)
        W /= W.norm(dim=0, keepdim=True) + 1e-8

    def encode_pre(self, x):
        return self.encoder(x - self.pre_bias)

    def forward(self, x):
        pre = self.encode_pre(x)
        z = topk_mask(pre, self.k)
        recon = self.decoder(z) + self.pre_bias
        return recon, z, pre


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load all activations for the requested hook into CPU memory (up to ~5GB for 250K*40*128*fp16)
    with h5py.File(args.acts, "r") as f:
        X = np.asarray(f["values"][:], dtype=np.float32)
    print(f"loaded activations: {X.shape}", flush=True)
    d_in = X.shape[1]
    n_latents = args.expansion * d_in
    k = args.k
    k_aux = min(args.k_aux, n_latents - 1)

    # train/val split (90/10 by sample index)
    N = X.shape[0]
    perm = np.random.default_rng(0).permutation(N)
    nval = N // 10
    val_idx = perm[:nval]; tr_idx = perm[nval:]
    X_tr = torch.from_numpy(X[tr_idx])
    X_va = torch.from_numpy(X[val_idx])
    var_x = float(X_tr.var().item())

    sae = TopKSAE(d_in, n_latents, k).to(device)
    # initialize pre_bias to median of x
    with torch.no_grad():
        sae.pre_bias.copy_(torch.from_numpy(np.median(X[:100_000], axis=0)).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    # dead-latent tracking: steps since last activation
    steps_since = torch.zeros(n_latents, dtype=torch.long, device=device)
    dead_window = args.dead_window

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.hook}_r{args.expansion}_k{k}"
    log = open(out_dir / f"{tag}_train.csv", "w")
    log.write("step,loss,mse,aux,l0,dead_frac,util_frac,val_mse\n")
    print(f"tag={tag}  d_in={d_in}  n_latents={n_latents}  k={k}", flush=True)

    t0 = time.time()
    bs = args.batch_size
    steps_per_epoch = max(1, len(tr_idx) // bs)
    total_steps = args.epochs * steps_per_epoch
    step = 0
    for ep in range(args.epochs):
        order = torch.randperm(len(tr_idx))
        for i in range(0, len(tr_idx) - bs + 1, bs):
            batch = X_tr[order[i:i+bs]].to(device)
            recon, z, pre = sae(batch)
            mse = F.mse_loss(recon, batch)

            # update dead-latent tracker
            any_nonzero = (z.abs() > 0).any(dim=0)  # (n_latents,)
            steps_since[any_nonzero] = 0
            steps_since[~any_nonzero] += 1
            dead_mask = steps_since > dead_window

            # AuxK: reconstruct residual using top-k_aux dead latents
            aux = torch.tensor(0.0, device=device)
            if dead_mask.any():
                z_aux = topk_on_restricted(pre, k_aux, dead_mask)
                resid = batch - recon.detach()
                e_hat = sae.decoder(z_aux)
                aux = F.mse_loss(e_hat, resid)

            loss = mse + (1.0 / 32.0) * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            if step % args.log_every == 0:
                with torch.no_grad():
                    l0 = float((z.abs() > 0).float().sum(dim=-1).mean().item())
                    dead_frac = float(dead_mask.float().mean().item())
                    util_frac = float((steps_since <= dead_window).float().mean().item())
                    # val MSE on fixed 10k sample
                    vmask = torch.randperm(len(X_va))[:10_000]
                    vb = X_va[vmask].to(device)
                    vrec, _, _ = sae(vb)
                    vmse = float(F.mse_loss(vrec, vb).item())
                log.write(f"{step},{loss.item():.6f},{mse.item():.6f},{float(aux):.6f},"
                          f"{l0:.2f},{dead_frac:.4f},{util_frac:.4f},{vmse:.6f}\n")
                log.flush()
                if step % (args.log_every * 10) == 0:
                    print(f"[{tag}] step={step}/{total_steps} mse={mse.item():.4f} "
                          f"aux={float(aux):.4f} L0={l0:.1f} dead={dead_frac:.2%} "
                          f"val_mse={vmse:.4f} var_x={var_x:.4f} "
                          f"elapsed={time.time()-t0:.1f}s", flush=True)
            step += 1
    log.close()

    # final val MSE
    with torch.no_grad():
        parts = []
        for i in range(0, len(X_va), bs):
            b = X_va[i:i+bs].to(device)
            r, _, _ = sae(b)
            parts.append(F.mse_loss(r, b).item() * b.size(0))
        val_mse = float(sum(parts) / len(X_va))

    torch.save({
        "model": sae.state_dict(),
        "d_in": d_in, "n_latents": n_latents, "k": k,
        "expansion": args.expansion, "hook": args.hook,
        "val_mse": val_mse, "var_x": var_x,
    }, out_dir / f"{tag}.pt")
    (out_dir / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "hook": args.hook, "r": args.expansion, "k": k,
        "val_mse": val_mse, "var_x": var_x,
        "val_mse_over_var": val_mse / max(var_x, 1e-8),
    }, indent=2))
    print(f"[{tag}] DONE val_mse={val_mse:.4f} var_x={var_x:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts", required=True, help="path to {stage}_activations.h5")
    p.add_argument("--hook", required=True, help="stage1/stage2/stage3")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--expansion", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--k_aux", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dead_window", type=int, default=None,
                   help="dead = no activation for this many training steps; "
                        "default ~1 full epoch")
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()
    if args.dead_window is None:
        # approximate 1 epoch; set dynamically after we know steps_per_epoch
        args.dead_window = 10_000
    train(args)
