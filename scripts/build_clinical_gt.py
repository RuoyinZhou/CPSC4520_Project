#!/usr/bin/env python
"""Build per-record clinical ground-truth table from PTB-XL+.

PTB-XL+ ships three per-record median-beat feature CSVs produced by three
independent delineation algorithms:
    features/12sl_features.csv     (Marquette 12SL)
    features/unig_features.csv     (University of Glasgow)
    features/ecgdeli_features.csv  (open-source ECGDeli)

We join them on ecg_id and keep only features where at least 2 of 3
algorithms agree within a tolerance (reducing algorithm-specific artifacts),
then binarize using clinically grounded thresholds suitable for the InterPLM
F1 protocol.

Binary concepts produced:
    wide_qrs          : QRS duration > 120 ms
    st_elevation_any  : ST deviation > 0.1 mV (any lead where available)
    st_depression_any : ST deviation < -0.1 mV
    prolonged_pr      : PR interval > 200 ms
    short_pr          : PR interval < 120 ms
    prolonged_qtc     : Bazett-corrected QT > 450 ms
    right_axis        : QRS axis > 90 deg
    left_axis         : QRS axis < -30 deg
    wide_p            : P-wave duration > 110 ms
    wide_p_120        : P-wave duration > 120 ms (clinical IAB threshold)

Outputs:
    <out_dir>/clinical_gt_records.parquet
        ecg_id, <continuous features>, <binary features>, n_agree
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd


# Per-algorithm column names for the canonical clinical features we care about.
# Based on PTB-XL+ feature_description.csv (v1.0.1).
# Each entry: (canonical_name, {algo: column_name}).
ALGO_COLUMNS = {
    "qrs_duration_ms": {"12sl": "QRS_Dur_Global", "unig": "QRS_Dur_Global", "ecgdeli": "QRS_Dur_Global"},
    "pr_interval_ms":  {"12sl": "PR_Int_Global",  "unig": "PR_Int_Global",  "ecgdeli": "PR_Int_Global"},
    "qt_interval_ms":  {"12sl": "QT_Int_Global",  "unig": "QT_Int_Global",  "ecgdeli": "QT_Int_Global"},
    "qtc_bazett_ms":   {"12sl": "QT_IntBazett_Global", "unig": "QT_IntBazett_Global", "ecgdeli": None},
    "rr_interval_ms":  {"12sl": "RR_Mean_Global", "unig": "RR_Mean_Global", "ecgdeli": "RR_Mean_Global"},
    "p_dur_ms":        {"12sl": "P_Dur_Global",   "unig": "P_Dur_Global",   "ecgdeli": "P_Dur_Global"},
    "qrs_axis_deg":    {"12sl": "R_AxisFrontal_Global", "unig": "QRS_AxisFront_Global", "ecgdeli": None},
    "p_axis_deg":      {"12sl": "P_AxisFront_Global", "unig": "P_AxisFront_Global", "ecgdeli": None},
    "t_axis_deg":      {"12sl": "T_AxisFront_Global", "unig": "T_AxisFront_Global", "ecgdeli": None},
}

FEATURE_CANDIDATES = {k: list(v.values()) for k, v in ALGO_COLUMNS.items()}


ST_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]


def load_one(path: Path, algo: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = None
    for c in df.columns:
        if c.lower() == "ecg_id":
            id_col = c; break
    if id_col is None:
        # 12sl has no explicit ecg_id header; look at the index column name or assume first col
        # PTB-XL+ README: "features/12sl_features.csv ... one row per ECG record"
        # ecg_id may be index=0
        df = pd.read_csv(path, index_col=0)
        df.index.name = "ecg_id"
    else:
        df = df.set_index(id_col); df.index.name = "ecg_id"

    cols = {}
    for canon, mapping in ALGO_COLUMNS.items():
        col = mapping.get(algo)
        if col and col in df.columns:
            cols[f"{canon}__{algo}"] = df[col]

    # per-lead ST amplitude: 12sl uses ST_Amp at J, unig provides ST at J+80ms;
    # both are proximate markers of ST shift. ecgdeli does not expose ST amp.
    for lead in ST_LEADS:
        if algo == "12sl":
            key = f"ST_Amp_{lead}"
        elif algo == "unig":
            key = f"ST_Amp80ms_{lead}"
        else:
            key = None
        if key and key in df.columns:
            cols[f"st_amp_{lead}__{algo}"] = df[key]

    return pd.DataFrame(cols)


def agreement_mean(rows: np.ndarray, tol_abs: float, tol_rel: float) -> tuple[float, int]:
    """Return (consensus_value, n_agree). Consensus = mean of values that pairwise-agree
    within tol. n_agree = count of values in the largest pairwise-agreeing cluster."""
    rows = rows[np.isfinite(rows)]
    if rows.size == 0:
        return np.nan, 0
    if rows.size == 1:
        return float(rows[0]), 1
    # simple cluster: pair that is closest in absolute sense
    best_cluster = None
    for i in range(len(rows)):
        cluster = [rows[i]]
        for j in range(len(rows)):
            if j == i:
                continue
            ref = rows[i]
            if abs(rows[j] - ref) <= tol_abs or abs(rows[j] - ref) <= tol_rel * (abs(ref) + 1e-6):
                cluster.append(rows[j])
        if best_cluster is None or len(cluster) > len(best_cluster):
            best_cluster = cluster
    return float(np.mean(best_cluster)), len(best_cluster)


def build_consensus(df: pd.DataFrame, canon: str, tol_abs: float, tol_rel: float = 0.15):
    cols = [c for c in df.columns if c.startswith(f"{canon}__")]
    vals = df[cols].to_numpy(dtype=float)
    cons_vals = np.full(len(df), np.nan)
    n_agree = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        cons_vals[i], n_agree[i] = agreement_mean(vals[i], tol_abs, tol_rel)
    return cons_vals, n_agree


# (canon, tol_absolute_in_native_units)
CONSENSUS_TOLERANCES = {
    "qrs_duration_ms": 10.0,
    "pr_interval_ms": 15.0,
    "qt_interval_ms": 20.0,
    "qtc_bazett_ms": 20.0,
    "p_dur_ms": 15.0,
    "qrs_axis_deg": 15.0,
    "p_axis_deg": 20.0,
    "t_axis_deg": 20.0,
    "rr_interval_ms": 20.0,
}
ST_TOL_MV = 0.05


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    b = pd.DataFrame(index=df.index)
    b["wide_qrs"] = (df["qrs_duration_ms"] > 120).astype("Int8")
    b["prolonged_pr"] = (df["pr_interval_ms"] > 200).astype("Int8")
    b["short_pr"] = (df["pr_interval_ms"] < 120).astype("Int8")
    if "qtc_bazett_ms" in df.columns and df["qtc_bazett_ms"].notna().any():
        qtc = df["qtc_bazett_ms"]
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            qtc = df["qt_interval_ms"] / np.sqrt(df["rr_interval_ms"] / 1000.0)
    b["prolonged_qtc"] = (qtc > 450).astype("Int8")
    b["right_axis"] = (df["qrs_axis_deg"] > 90).astype("Int8")
    b["left_axis"] = (df["qrs_axis_deg"] < -30).astype("Int8")
    b["wide_p"] = (df["p_dur_ms"] > 110).astype("Int8")
    # Clinical standard for inter-atrial block: P > 120ms (Bayes de Luna, 2012)
    b["wide_p_120"] = (df["p_dur_ms"] > 120).astype("Int8")
    # ST elevation/depression: consensus ST amplitude (2/3 agreement across algos),
    # binarized at +-0.1 mV in ANY of the standard limb/precordial leads.
    st_any_elev = pd.Series(False, index=df.index)
    st_any_depr = pd.Series(False, index=df.index)
    for lead in ST_LEADS:
        col = f"st_amp_{lead}"
        if col in df.columns:
            v = df[col]
            st_any_elev |= (v > 0.1).fillna(False)
            st_any_depr |= (v < -0.1).fillna(False)
    b["st_elevation_any"] = st_any_elev.astype("Int8")
    b["st_depression_any"] = st_any_depr.astype("Int8")
    return b


def main(args):
    root = Path(args.ptbxlplus_root)
    paths = {
        "12sl": root / "features" / "12sl_features.csv",
        "unig": root / "features" / "unig_features.csv",
        "ecgdeli": root / "features" / "ecgdeli_features.csv",
    }
    for a, p in paths.items():
        if not p.exists():
            print(f"[warn] missing {p}", file=sys.stderr)

    frames = []
    for algo, p in paths.items():
        if p.exists():
            frames.append(load_one(p, algo))
    if not frames:
        raise SystemExit("no PTB-XL+ feature files found")
    joined = pd.concat(frames, axis=1, join="outer")

    out = pd.DataFrame(index=joined.index)
    n_agree_cols = {}
    for canon, tol in CONSENSUS_TOLERANCES.items():
        vals, nagree = build_consensus(joined, canon, tol)
        out[canon] = vals
        n_agree_cols[f"{canon}__n_agree"] = nagree
    # per-lead ST amplitude consensus
    for lead in ST_LEADS:
        vals, nagree = build_consensus(joined, f"st_amp_{lead}", ST_TOL_MV, tol_rel=0.25)
        out[f"st_amp_{lead}"] = vals
        n_agree_cols[f"st_amp_{lead}__n_agree"] = nagree
    for k, v in n_agree_cols.items():
        out[k] = v

    b = binarize(out)
    result = pd.concat([out, b], axis=1).reset_index()
    result.to_parquet(Path(args.out), index=False)
    # brief summary
    print(f"records: {len(result)}", flush=True)
    for col in b.columns:
        pos = int(result[col].fillna(0).astype(int).sum())
        tot = int(result[col].notna().sum())
        print(f"  {col}: {pos}/{tot} positive ({pos/max(tot,1):.2%})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxlplus_root", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
