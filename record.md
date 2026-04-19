# Pipeline Record

Goal: open the black box of an ECG classifier using Sparse Autoencoders (SAEs).
Do its internal features match what cardiologists look for?

## Steps

1. **Download data** — PTB-XL (21k 10-s 12-lead ECGs, 5 superclasses: NORM/MI/STTC/CD/HYP) + PTB-XL+ (clinical features from 3 algorithms).

5 superclasses: 5 大类诊断标签:
NORM = 正常
MI = 心肌梗死(Myocardial Infarction)
STTC = ST/T 段改变(ST-T Changes)
CD = 传导异常(Conduction Disturbance)
HYP = 肥厚(Hypertrophy)

2. **Train classifier** — `resnet1d_wang` (Wang 2017 / Strodthoff 2021 PTB-XL benchmark architecture: stem + 3 residual stages, 128 channels) on folds 1–8 (train), 9 (val), 10 (test). Test macro-AUC = 0.9051.

3. **Extract activations** — Run the classifier on every record, detect R-peaks, save ±0.4 s windows of stage1/stage2/stage3 outputs (128-d per timestep) to HDF5.

   *Why R-peak windows instead of the full 10 s?* A 10 s ECG contains 8–15 heartbeats at arbitrary time offsets. Feeding the whole signal to the SAE would force it to learn position-dependent noise rather than beat-level morphology. Aligning each sample to the R-peak (beat center) yields one complete P-QRS-T cycle per sample, all in the same reference frame — so the SAE can learn "what this beat looks like" instead of "where the beat happened to fall in the window." It also multiplies the effective dataset size ~10× and matches how clinicians actually read ECGs (beat by beat).

4. **Train 36 SAEs** — Sweep 3 hooks × 4 expansion ratios r ∈ {4, 8, 16, 32} × 3 sparsity levels k ∈ {8, 16, 32}. Each SAE: `z = TopK_k(W_enc(x − b))`, `x̂ = W_dec z + b`, with AuxK dead-latent revival.

5. **Build clinical ground truth** — Merge PTB-XL+ algorithm outputs with ≥2/3 agreement. Binarize into concepts (wide_qrs, left_axis, prolonged_qtc, ST elevation, etc.).

6. **Feature interpretation (InterPLM F1 protocol)** — For each (latent, concept) pair: max-pool activations per beat, sweep thresholds, measure F1 on val/test. Monosemantic = val_F1 > 0.5 AND test_F1 > 0.5.

7. **Causal ablation** — For shortlisted latents: forward-pass three times (clean / SAE-reconstructed / latent-zeroed). Δ_causal = AUC_recon − AUC_abl isolates each latent's contribution. Includes bootstrap CI, grouped ablation (top-K together), and random-latent null baseline.


## Headline Result

Best config: **stage2_r32_k32** (4096 latents).
- 314 monosemantic latents (7.7%) — but 300 track only "NORM" (class imbalance artifact).
- 14 track specific clinical concepts: 9 left_axis, 3 MI, 2 CD.
- **left_axis** is the cleanest (F1 = 0.61).
- MI/CD monosemantic latents appear only at r ≥ 16 — minority concepts need more capacity.
- Reconstruction tax ≤ 0.003 AUC — SAE replacement is near-lossless, so ablation deltas are causal.
- Negative finding: prolonged_pr (F1=0.26), wide_qrs (0.47), prolonged_qtc (0.47) — clinically important but not cleanly recovered.

## Takeaway

SAEs can surface some real clinical concepts (left_axis) and reveal scaling laws for minority classes, but most "monosemantic" latents are dominated by the majority class, and several textbook ECG markers are not recovered. Useful but limited.
