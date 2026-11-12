"""Build consolidated TGAT v3 + controlled ablation + baseline research evidence package.

Outputs under results/research_validation_v3/final_paper_evidence/:
  - v3_three_seed_summary.json / .csv
  - controlled_v3_component_multiseed_summary.csv / .json
  - controlled_v3_ablations_all_runs.csv / .json
  - controlled_v3_ablations_aggregated.csv
  - baseline_vs_tgat_v3_comparison.csv / .json
  - historical_v1_and_v2_audit.csv / .json
  - confusion_matrices_all.json / .csv
  - non_completed_experiments_log.json
  - reproducibility_manifest.json / .csv
  - final_paper_evidence_master.json
  - plots/figA_roc_and_pr_baselines_vs_tgat_v3.png
  - plots/figB_controlled_v3_component_ablations_multiseed.png
  - plots/figC_controlled_v3_k_sweep_and_architecture.png
  - plots/figD_threshold_and_confusion_matrices.png
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

REPO = Path(r"c:\Users\duxdh\Downloads\tgat-research-validation")
MAIN_APP = Path(r"c:\Users\duxdh\Downloads\tgat")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models.common.evaluation import classification_metrics
from src.analysis.threshold_analysis import threshold_grid, sweep

V3_DIR = REPO / "results" / "research_validation_v3"
V2_DIR = REPO / "results" / "research_validation"
OUT_DIR = V3_DIR / "final_paper_evidence"
PLOTS_DIR = OUT_DIR / "plots"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else "Is Laundering"
    score_col = "score" if "score" in df.columns else "probability"
    y = df[label_col].to_numpy(dtype=np.int8)
    s = np.nan_to_num(df[score_col].to_numpy(dtype=float), nan=0.0)
    return y, s


def exact_pr_max_f1(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    precisions, recalls, thresholds = precision_recall_curve(y, s)
    p = precisions[:-1]
    r = recalls[:-1]
    denom = p + r
    f1s = np.divide(2.0 * p * r, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    idx = int(np.argmax(f1s))
    th = float(thresholds[idx])
    return classification_metrics(y, s, threshold=th, k_values=(100, 500, 1000))


def evaluate_from_predictions(val_csv: Path, test_csv: Path, primary_tau: float | None = None) -> dict[str, Any]:
    yv, sv = load_predictions(val_csv)
    yt, st = load_predictions(test_csv)
    k_vals = (50, 100, 250, 500, 1000, 2500, 5000)

    grid = threshold_grid(sv, points=1000)
    val_sweep = sweep(yv, sv, grid)
    best_row = val_sweep.loc[val_sweep["f1"].idxmax()]
    own_tau = float(best_row["threshold"])

    exact_val = exact_pr_max_f1(yv, sv)
    exact_tau = float(exact_val["threshold"])

    res = {
        "val_at_0_5": classification_metrics(yv, sv, threshold=0.5, k_values=k_vals),
        "test_at_0_5": classification_metrics(yt, st, threshold=0.5, k_values=k_vals),
        "own_max_f1_threshold_grid1000": own_tau,
        "val_at_own_max_f1": classification_metrics(yv, sv, threshold=own_tau, k_values=k_vals),
        "test_at_own_max_f1": classification_metrics(yt, st, threshold=own_tau, k_values=k_vals),
        "exact_max_f1_threshold_pr_curve": exact_tau,
        "val_at_exact_max_f1": classification_metrics(yv, sv, threshold=exact_tau, k_values=k_vals),
        "test_at_exact_max_f1": classification_metrics(yt, st, threshold=exact_tau, k_values=k_vals),
    }
    if primary_tau is not None:
        res["val_at_primary_v3_tau"] = classification_metrics(yv, sv, threshold=primary_tau, k_values=k_vals)
        res["test_at_primary_v3_tau"] = classification_metrics(yt, st, threshold=primary_tau, k_values=k_vals)
    return res


def extract_tgat_run_record(
    run_id: str,
    group: str,
    ablation_family: str,
    label: str,
    run_dir: Path,
    provenance: str,
    primary_tau: float,
    legacy_res: int = 1,
) -> dict[str, Any]:
    metrics_json = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.json"
    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    tlog_path = run_dir / "training_log.json"
    tlog = json.loads(tlog_path.read_text(encoding="utf-8")) if tlog_path.exists() else None
    val_csv = run_dir / "validation_predictions.csv"
    test_csv = run_dir / "test_predictions.csv"

    eval_data = evaluate_from_predictions(val_csv, test_csv, primary_tau=primary_tau) if (val_csv.exists() and test_csv.exists()) else None
    stab = metrics_json.get("training_stability", {})
    v = metrics_json.get("validation", {})
    t = metrics_json.get("test", {})
    te_grads = stab.get("epoch_te_grad_norm_means") or []
    ep_losses = [x for x in (stab.get("epoch_losses") or ([] if not tlog else [e.get("loss") for e in tlog])) if x is not None]
    ep_gmeans = [x for x in (stab.get("epoch_grad_norm_means") or ([] if not tlog else [e.get("pre_clip_grad_norm", {}).get("mean") for e in tlog])) if x is not None]
    ep_gmaxes = [x for x in (stab.get("epoch_grad_norm_maxes") or ([] if not tlog else [e.get("pre_clip_grad_norm", {}).get("max") for e in tlog])) if x is not None]

    row = {
        "run_id": run_id,
        "group": group,
        "ablation_family": ablation_family,
        "label": label,
        "provenance_status": provenance,
        "dataset": cfg_json.get("data_dir", metrics_json.get("data_dir", "data/processed/temporal_gnn_v2_1s" if legacy_res == 1 else "data/processed/temporal_gnn")),
        "timestamp_resolution_s": metrics_json.get("timestamp_resolution_seconds", legacy_res),
        "time_encoding_mode": metrics_json.get("time_encoding_mode", "raw_seconds" if legacy_res == 1 else "legacy_raw_seconds"),
        "ablation": metrics_json.get("ablation", cfg_json.get("ablation", "none")),
        "seed": metrics_json.get("seed", cfg_json.get("seed", 42)),
        "hidden_dim": cfg_json.get("hidden_dim", 32),
        "num_layers": cfg_json.get("num_layers", 2),
        "num_heads": cfg_json.get("num_heads", 2),
        "dropout": cfg_json.get("dropout", 0.2),
        "neighbor_k": metrics_json.get("neighbor_k", cfg_json.get("neighbor_k", 10)),
        "max_history": metrics_json.get("max_history", cfg_json.get("max_history", 10)),
        "feature_count": metrics_json.get("feature_count", 26),
        "model_parameters": metrics_json.get("model_parameters"),
        "device": metrics_json.get("device"),
        "selected_epoch": metrics_json.get("selected_epoch"),
        "training_seconds": metrics_json.get("training_seconds"),
        "val_roc_auc": v.get("roc_auc"),
        "val_pr_auc": v.get("pr_auc"),
        "test_roc_auc": t.get("roc_auc"),
        "test_pr_auc": t.get("pr_auc"),
        "test_p_at_100": t.get("precision_at_100"),
        "test_r_at_100": t.get("recall_at_100"),
        "test_p_at_500": t.get("precision_at_500"),
        "test_r_at_500": t.get("recall_at_500"),
        "test_p_at_1000": t.get("precision_at_1000"),
        "test_r_at_1000": t.get("recall_at_1000"),
        "epoch_1_loss": ep_losses[0] if len(ep_losses) >= 1 else None,
        "epoch_3_loss": ep_losses[-1] if len(ep_losses) >= 1 else None,
        "grad_norm_mean_last": ep_gmeans[-1] if ep_gmeans else None,
        "grad_norm_max_overall": max(ep_gmaxes) if ep_gmaxes else None,
        "overall_clip_fraction": stab.get("overall_clip_fraction"),
        "te_grad_mean_last": te_grads[-1] if te_grads else None,
        "checkpoint_path": str(run_dir / "model.pt") if (run_dir / "model.pt").exists() else None,
        "config_path": str(cfg_path) if cfg_path.exists() else None,
        "scaler_path": str(run_dir / "scaler_params.json") if (run_dir / "scaler_params.json").exists() else None,
        "metrics_path": str(run_dir / "metrics.json"),
        "training_log_path": str(tlog_path) if tlog_path.exists() else None,
        "val_predictions_path": str(val_csv) if val_csv.exists() else None,
        "test_predictions_path": str(test_csv) if test_csv.exists() else None,
    }
    if eval_data is not None:
        tf = eval_data["test_at_own_max_f1"]
        t05 = eval_data["test_at_0_5"]
        tef = eval_data["test_at_exact_max_f1"]
        row.update({
            "own_val_frozen_tau": eval_data["own_max_f1_threshold_grid1000"],
            "test_frozen_precision": tf["precision"],
            "test_frozen_recall": tf["recall"],
            "test_frozen_f1": tf["f1"],
            "test_frozen_fpr": tf["false_positive_rate"],
            "test_frozen_tp": tf["true_positives"],
            "test_frozen_fp": tf["false_positives"],
            "test_frozen_tn": tf["true_negatives"],
            "test_frozen_fn": tf["false_negatives"],
            "exact_val_frozen_tau": eval_data["exact_max_f1_threshold_pr_curve"],
            "test_exact_frozen_f1": tef["f1"],
            "test_050_precision": t05["precision"],
            "test_050_recall": t05["recall"],
            "test_050_f1": t05["f1"],
            "test_050_fpr": t05["false_positive_rate"],
            "test_050_tp": t05["true_positives"],
            "test_050_fp": t05["false_positives"],
            "test_050_tn": t05["true_negatives"],
            "test_050_fn": t05["false_negatives"],
        })
        if "test_at_primary_v3_tau" in eval_data:
            tp = eval_data["test_at_primary_v3_tau"]
            row.update({
                "test_primary_tau_f1": tp["f1"],
                "test_primary_tau_precision": tp["precision"],
                "test_primary_tau_recall": tp["recall"],
                "test_primary_tau_tp": tp["true_positives"],
                "test_primary_tau_fp": tp["false_positives"],
            })
    return row


def agg_stats(rows: list[dict[str, Any]], metrics_keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for k in metrics_keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": vals,
            }
    return out


def build_all() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_th = json.loads((V3_DIR / "threshold_analysis" / "frozen_threshold.json").read_text(encoding="utf-8"))
    primary_tau = float(frozen_th["threshold"])

    # 1. Full TGAT v3 (3 seeds) + Controlled v3 Ablations
    v3_core_specs = [
        ("tgat_v3_seed42", "Full TGAT v3", "full_v3", "Full TGAT v3 (seed 42)", V3_DIR / "corrected_tgat_v3_seed42"),
        ("tgat_v3_seed43", "Full TGAT v3", "full_v3", "Full TGAT v3 (seed 43)", V3_DIR / "corrected_tgat_v3_seed43"),
        ("tgat_v3_seed44", "Full TGAT v3", "full_v3", "Full TGAT v3 (seed 44)", V3_DIR / "corrected_tgat_v3_seed44"),
    ]

    v3_ablation_specs = [
        ("v3_no_time_encoding_s42", "Core Component Ablation (3-Seed)", "no_time_encoding", "No Time Encoding (seed 42)", V3_DIR / "ablation_no_time_encoding"),
        ("v3_no_time_encoding_s43", "Core Component Ablation (3-Seed)", "no_time_encoding", "No Time Encoding (seed 43)", V3_DIR / "ablations_multiseed" / "no_time_encoding_seed43"),
        ("v3_no_time_encoding_s44", "Core Component Ablation (3-Seed)", "no_time_encoding", "No Time Encoding (seed 44)", V3_DIR / "ablations_multiseed" / "no_time_encoding_seed44"),
        ("v3_no_neighbors_s42", "Core Component Ablation (3-Seed)", "no_neighbors", "No Neighbors (seed 42)", V3_DIR / "ablations" / "b_no_neighbors"),
        ("v3_no_neighbors_s43", "Core Component Ablation (3-Seed)", "no_neighbors", "No Neighbors (seed 43)", V3_DIR / "ablations_multiseed" / "b_no_neighbors_seed43"),
        ("v3_no_neighbors_s44", "Core Component Ablation (3-Seed)", "no_neighbors", "No Neighbors (seed 44)", V3_DIR / "ablations_multiseed" / "b_no_neighbors_seed44"),
        ("v3_no_temporal_features_s42", "Core Component Ablation (3-Seed)", "no_temporal_features", "No Temporal Features (seed 42)", V3_DIR / "ablations" / "a_no_temporal_features"),
        ("v3_no_temporal_features_s43", "Core Component Ablation (3-Seed)", "no_temporal_features", "No Temporal Features (seed 43)", V3_DIR / "ablations_multiseed" / "a_no_temporal_features_seed43"),
        ("v3_no_temporal_features_s44", "Core Component Ablation (3-Seed)", "no_temporal_features", "No Temporal Features (seed 44)", V3_DIR / "ablations_multiseed" / "a_no_temporal_features_seed44"),
        ("v3_k_1_s42", "Neighbor Budget Sweep (Seed 42)", "k_1_hist20", "K=1, hist=20 (seed 42)", V3_DIR / "ablations" / "c_neighbor_k" / "k_1"),
        ("v3_k_5_s42", "Neighbor Budget Sweep (Seed 42)", "k_5_hist20", "K=5, hist=20 (seed 42)", V3_DIR / "ablations" / "c_neighbor_k" / "k_5"),
        ("v3_k_10_s42", "Neighbor Budget Sweep (Seed 42)", "k_10_hist20", "K=10, hist=20 (seed 42)", V3_DIR / "ablations" / "c_neighbor_k" / "k_10"),
        ("v3_k_20_s42", "Neighbor Budget Sweep (Seed 42)", "k_20_hist20", "K=20, hist=20 (seed 42)", V3_DIR / "ablations" / "c_neighbor_k" / "k_20"),
        ("v3_e1_one_layer_s42", "Architecture Sensitivity (Seed 42)", "e1_one_layer", "1 Attention Layer (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e1_one_layer"),
        ("v3_e2_one_head_s42", "Architecture Sensitivity (Seed 42)", "e2_one_head", "1 Attention Head (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e2_one_head"),
        ("v3_e3_hidden_16_s42", "Architecture Sensitivity (Seed 42)", "e3_hidden_16", "Hidden Dim = 16 (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e3_hidden_16"),
        ("v3_e4_hidden_64_s42", "Architecture Sensitivity (Seed 42)", "e4_hidden_64", "Hidden Dim = 64 (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e4_hidden_64"),
        ("v3_e5_dropout_0_s42", "Regularization Sensitivity (Seed 42)", "e5_dropout_0", "Dropout = 0.0 (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e5_dropout_0"),
        ("v3_e6_dropout_04_s42", "Regularization Sensitivity (Seed 42)", "e6_dropout_04", "Dropout = 0.4 (seed 42)", V3_DIR / "ablations" / "e_architecture" / "e6_dropout_04"),
    ]

    v3_rows: list[dict[str, Any]] = []
    for rid, grp, fam, lbl, rdir in v3_core_specs + v3_ablation_specs:
        if not (rdir / "metrics.json").exists():
            print(f"[WARN] Missing controlled v3 run: {rid} at {rdir}")
            continue
        v3_rows.append(
            extract_tgat_run_record(rid, grp, fam, lbl, rdir, "Newly Trained (v3, 1s log1p)", primary_tau, legacy_res=1)
        )

    v3_df = pd.DataFrame(v3_rows)
    v3_df.to_csv(OUT_DIR / "controlled_v3_ablations_all_runs.csv", index=False)
    (OUT_DIR / "controlled_v3_ablations_all_runs.json").write_text(json.dumps(v3_rows, indent=2), encoding="utf-8")

    # Aggregated per-family table
    agg_keys = [
        "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc",
        "test_p_at_100", "test_p_at_500", "test_p_at_1000",
        "test_r_at_100", "test_r_at_500", "test_r_at_1000",
        "own_val_frozen_tau", "test_frozen_precision", "test_frozen_recall", "test_frozen_f1", "test_frozen_fpr",
        "test_frozen_tp", "test_frozen_fp", "test_frozen_tn", "test_frozen_fn",
        "test_050_precision", "test_050_recall", "test_050_f1", "test_050_fpr", "test_050_tp", "test_050_fp",
        "training_seconds", "overall_clip_fraction", "grad_norm_mean_last", "grad_norm_max_overall",
    ]
    fam_order = [
        "full_v3", "no_time_encoding", "no_neighbors", "no_temporal_features",
        "k_1_hist20", "k_5_hist20", "k_10_hist20", "k_20_hist20",
        "e1_one_layer", "e2_one_head", "e3_hidden_16", "e4_hidden_64", "e5_dropout_0", "e6_dropout_04",
    ]
    fam_labels = {
        "full_v3": "Full TGAT v3 (K=10, hist=10, L=2, H=2, d=32, p=0.2)",
        "no_time_encoding": "Ablation D: No Time Encoding",
        "no_neighbors": "Ablation B: No Neighbors (K=0)",
        "no_temporal_features": "Ablation A: No Temporal Behavioural Features (11 raw)",
        "k_1_hist20": "Ablation C1: Neighbor K=1 (hist=20)",
        "k_5_hist20": "Ablation C2: Neighbor K=5 (hist=20)",
        "k_10_hist20": "Ablation C3: Neighbor K=10 (hist=20)",
        "k_20_hist20": "Ablation C4: Neighbor K=20 (hist=20)",
        "e1_one_layer": "Ablation E1: 1 Attention Layer",
        "e2_one_head": "Ablation E2: 1 Attention Head",
        "e3_hidden_16": "Ablation E3: Hidden Dim = 16",
        "e4_hidden_64": "Ablation E4: Hidden Dim = 64",
        "e5_dropout_0": "Ablation E5: Dropout = 0.0",
        "e6_dropout_04": "Ablation E6: Dropout = 0.4",
    }
    agg_rows = []
    agg_dict: dict[str, Any] = {}
    for fam in fam_order:
        frows = [r for r in v3_rows if r["ablation_family"] == fam]
        if not frows:
            continue
        st = agg_stats(frows, agg_keys)
        agg_dict[fam] = st
        flat = {
            "ablation_family": fam,
            "config_label": fam_labels[fam],
            "group": frows[0]["group"],
            "n_seeds": st["n_seeds"],
            "seeds": ",".join(str(s) for s in st["seeds"]),
            "model_parameters": frows[0]["model_parameters"],
        }
        for k in agg_keys:
            if k in st:
                flat[f"{k}_mean"] = st[k]["mean"]
                flat[f"{k}_std"] = st[k]["std"]
        agg_rows.append(flat)

    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "controlled_v3_ablations_aggregated.csv", index=False)
    (OUT_DIR / "controlled_v3_component_multiseed_summary.json").write_text(
        json.dumps({k: agg_dict[k] for k in ["full_v3", "no_time_encoding", "no_neighbors", "no_temporal_features"] if k in agg_dict}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "v3_three_seed_summary.json").write_text(
        json.dumps({"per_seed": [r for r in v3_rows if r["ablation_family"] == "full_v3"], "aggregate": agg_dict.get("full_v3")}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([r for r in v3_rows if r["ablation_family"] == "full_v3"]).to_csv(OUT_DIR / "v3_three_seed_summary.csv", index=False)

    # 2. Baselines + TGAT v3 vs Historical v2 comparison (uniform validation-frozen max_f1 on all prediction CSVs!)
    baseline_specs = [
        {
            "model_id": "logistic_regression",
            "model_name": "Logistic Regression (SGD)",
            "model_type": "Linear Tabular Baseline",
            "provenance_status": "Historical Reference Baseline (modeling_splits, 53 features, seed 42)",
            "dir": MAIN_APP / "results" / "baselines" / "logistic_regression",
            "val_csv": MAIN_APP / "results" / "baselines" / "logistic_regression" / "predictions_validation.csv",
            "test_csv": MAIN_APP / "results" / "baselines" / "logistic_regression" / "predictions_test.csv",
            "cfg_path": MAIN_APP / "results" / "baselines" / "logistic_regression" / "config.json",
            "ckpt_path": MAIN_APP / "results" / "baselines" / "logistic_regression" / "model.joblib",
            "train_sec_path": ("training", "training_seconds"),
            "feature_count": 53,
        },
        {
            "model_id": "static_xgboost",
            "model_name": "Static XGBoost",
            "model_type": "Tree Tabular Baseline (No Graph/History)",
            "provenance_status": "Historical Reference Baseline (modeling_splits, 19 features, seed 42)",
            "dir": MAIN_APP / "results" / "baselines" / "xgboost",
            "val_csv": MAIN_APP / "results" / "baselines" / "xgboost" / "predictions_validation.csv",
            "test_csv": MAIN_APP / "results" / "baselines" / "xgboost" / "predictions_test.csv",
            "cfg_path": MAIN_APP / "results" / "baselines" / "xgboost" / "config.json",
            "ckpt_path": MAIN_APP / "results" / "baselines" / "xgboost" / "model.json",
            "train_sec_path": ("training", "training_seconds"),
            "feature_count": 19,
        },
        {
            "model_id": "static_gcn",
            "model_name": "Static GCN",
            "model_type": "Static Homogeneous Graph Baseline",
            "provenance_status": "Historical Reference Baseline (TRAIN static_gnn graph, zero-embed unseen)",
            "dir": MAIN_APP / "results" / "baselines" / "static_gcn",
            "val_csv": MAIN_APP / "results" / "baselines" / "static_gcn" / "validation_predictions.csv",
            "test_csv": MAIN_APP / "results" / "baselines" / "static_gcn" / "test_predictions.csv",
            "cfg_path": None,
            "ckpt_path": MAIN_APP / "results" / "baselines" / "static_gcn" / "model.pt",
            "train_sec_path": None,
            "feature_count": 11,
        },
        {
            "model_id": "temporal_xgboost_v1_1000s",
            "model_name": "Temporal XGBoost (v1, 1000s)",
            "model_type": "Temporal Stateful Tabular Baseline",
            "provenance_status": "Historical Baseline (v1 1,000s quantized timestamps, 27 features)",
            "dir": MAIN_APP / "results" / "baselines" / "temporal_xgboost",
            "val_csv": MAIN_APP / "results" / "baselines" / "temporal_xgboost" / "validation_predictions.csv",
            "test_csv": MAIN_APP / "results" / "baselines" / "temporal_xgboost" / "test_predictions.csv",
            "cfg_path": None,
            "ckpt_path": MAIN_APP / "results" / "baselines" / "temporal_xgboost" / "model.json",
            "train_sec_path": ("training_seconds",),
            "feature_count": 27,
        },
        {
            "model_id": "temporal_xgboost_v2_1s",
            "model_name": "Temporal XGBoost (v2, 1s)",
            "model_type": "Temporal Stateful Tabular Baseline",
            "provenance_status": "Controlled 1s Baseline (temporal_gnn_v2_1s, 27 features, seed 42)",
            "dir": V2_DIR / "temporal_xgboost",
            "val_csv": V2_DIR / "temporal_xgboost" / "validation_predictions.csv",
            "test_csv": V2_DIR / "temporal_xgboost" / "test_predictions.csv",
            "cfg_path": None,
            "ckpt_path": V2_DIR / "temporal_xgboost" / "model.json",
            "train_sec_path": ("training_seconds",),
            "feature_count": 27,
        },
    ]

    comp_rows: list[dict[str, Any]] = []
    for bs in baseline_specs:
        mjson = json.loads((bs["dir"] / "metrics.json").read_text(encoding="utf-8"))
        ev = evaluate_from_predictions(bs["val_csv"], bs["test_csv"])
        t_sec = None
        if bs["train_sec_path"]:
            cur = mjson
            for pkey in bs["train_sec_path"]:
                cur = cur.get(pkey, {}) if isinstance(cur, dict) else None
            t_sec = cur if isinstance(cur, (int, float)) else None
        tf = ev["test_at_own_max_f1"]
        t05 = ev["test_at_0_5"]
        v05 = ev["val_at_0_5"]
        comp_rows.append({
            "model_id": bs["model_id"],
            "model_name": bs["model_name"],
            "model_type": bs["model_type"],
            "provenance_status": bs["provenance_status"],
            "n_seeds": 1,
            "feature_count": bs["feature_count"],
            "training_seconds": t_sec,
            "val_roc_auc": v05["roc_auc"],
            "val_pr_auc": v05["pr_auc"],
            "test_roc_auc": t05["roc_auc"],
            "test_pr_auc": t05["pr_auc"],
            "test_p_at_100": t05["precision_at_100"],
            "test_p_at_500": t05["precision_at_500"],
            "test_p_at_1000": t05["precision_at_1000"],
            "test_r_at_100": t05["recall_at_100"],
            "test_r_at_500": t05["recall_at_500"],
            "test_r_at_1000": t05["recall_at_1000"],
            "own_val_frozen_tau": ev["own_max_f1_threshold_grid1000"],
            "test_frozen_precision": tf["precision"],
            "test_frozen_recall": tf["recall"],
            "test_frozen_f1": tf["f1"],
            "test_frozen_fpr": tf["false_positive_rate"],
            "test_frozen_tp": tf["true_positives"],
            "test_frozen_fp": tf["false_positives"],
            "test_frozen_tn": tf["true_negatives"],
            "test_frozen_fn": tf["false_negatives"],
            "test_050_precision": t05["precision"],
            "test_050_recall": t05["recall"],
            "test_050_f1": t05["f1"],
            "test_050_fpr": t05["false_positive_rate"],
            "test_050_tp": t05["true_positives"],
            "test_050_fp": t05["false_positives"],
            "metrics_path": str(bs["dir"] / "metrics.json"),
            "checkpoint_path": str(bs["ckpt_path"]),
            "val_predictions_path": str(bs["val_csv"]),
            "test_predictions_path": str(bs["test_csv"]),
        })

    # Add Historical TGAT v2 (3-seed mean + seed 42), Historical TGAT v1 (1000s), and Final TGAT v3 (3-seed mean + seed 42)
    v2_tgat_rows = [
        extract_tgat_run_record("tgat_v2_s42", "Historical v2", "tgat_v2", "TGAT v2 (1s raw_s, seed 42)", V2_DIR / "corrected_tgat", "Historical Reference (v2, 1s raw_s)", primary_tau, 1),
        extract_tgat_run_record("tgat_v2_s43", "Historical v2", "tgat_v2", "TGAT v2 (1s raw_s, seed 43)", V2_DIR / "corrected_tgat_seed43", "Historical Reference (v2, 1s raw_s)", primary_tau, 1),
        extract_tgat_run_record("tgat_v2_s44", "Historical v2", "tgat_v2", "TGAT v2 (1s raw_s, seed 44)", V2_DIR / "corrected_tgat_seed44", "Historical Reference (v2, 1s raw_s)", primary_tau, 1),
    ]
    v2_ms = agg_stats(v2_tgat_rows, agg_keys)
    comp_rows.append({
        "model_id": "tgat_v2_3seed_mean",
        "model_name": "TGAT v2 (1s, raw_seconds, 3-seed mean)",
        "model_type": "Temporal GNN (Unscaled TimeEncoder)",
        "provenance_status": "Historical Reference (v2, 1s raw_s, seeds 42/43/44)",
        "n_seeds": 3,
        "feature_count": 26,
        "training_seconds": v2_ms["training_seconds"]["mean"],
        "val_roc_auc": v2_ms["val_roc_auc"]["mean"],
        "val_pr_auc": v2_ms["val_pr_auc"]["mean"],
        "test_roc_auc": v2_ms["test_roc_auc"]["mean"],
        "test_pr_auc": v2_ms["test_pr_auc"]["mean"],
        "test_p_at_100": v2_ms["test_p_at_100"]["mean"],
        "test_p_at_500": v2_ms["test_p_at_500"]["mean"],
        "test_p_at_1000": v2_ms["test_p_at_1000"]["mean"],
        "test_r_at_100": v2_ms["test_r_at_100"]["mean"],
        "test_r_at_500": v2_ms["test_r_at_500"]["mean"],
        "test_r_at_1000": v2_ms["test_r_at_1000"]["mean"],
        "own_val_frozen_tau": v2_ms["own_val_frozen_tau"]["mean"],
        "test_frozen_precision": v2_ms["test_frozen_precision"]["mean"],
        "test_frozen_recall": v2_ms["test_frozen_recall"]["mean"],
        "test_frozen_f1": v2_ms["test_frozen_f1"]["mean"],
        "test_frozen_fpr": v2_ms["test_frozen_fpr"]["mean"],
        "test_frozen_tp": v2_ms["test_frozen_tp"]["mean"],
        "test_frozen_fp": v2_ms["test_frozen_fp"]["mean"],
        "test_frozen_tn": 282576.0 - v2_ms["test_frozen_fp"]["mean"],
        "test_frozen_fn": 301.0 - v2_ms["test_frozen_tp"]["mean"],
        "test_050_precision": v2_ms["test_050_precision"]["mean"],
        "test_050_recall": v2_ms["test_050_recall"]["mean"],
        "test_050_f1": v2_ms["test_050_f1"]["mean"],
        "test_050_fpr": v2_ms["test_050_fpr"]["mean"],
        "test_050_tp": v2_ms["test_050_tp"]["mean"],
        "test_050_fp": v2_ms["test_050_fp"]["mean"],
        "metrics_path": str(V2_DIR / "summary_metrics.json"),
        "checkpoint_path": str(V2_DIR / "corrected_tgat" / "model.pt"),
        "val_predictions_path": str(V2_DIR / "corrected_tgat" / "validation_predictions.csv"),
        "test_predictions_path": str(V2_DIR / "corrected_tgat" / "test_predictions.csv"),
    })

    # TGAT v3 primary Seed 42 AND 3-seed mean
    v3_s42 = [r for r in v3_rows if r["run_id"] == "tgat_v3_seed42"][0]
    v3_ms = agg_dict["full_v3"]
    comp_rows.append({
        "model_id": "tgat_v3_seed42_primary",
        "model_name": "TGAT v3 (1s, log1p_seconds, seed 42)",
        "model_type": "Temporal GNN (Corrected TimeEncoder)",
        "provenance_status": "Newly Trained Primary (v3, 1s log1p_seconds, seed 42)",
        "n_seeds": 1,
        "feature_count": 26,
        "training_seconds": v3_s42["training_seconds"],
        "val_roc_auc": v3_s42["val_roc_auc"],
        "val_pr_auc": v3_s42["val_pr_auc"],
        "test_roc_auc": v3_s42["test_roc_auc"],
        "test_pr_auc": v3_s42["test_pr_auc"],
        "test_p_at_100": v3_s42["test_p_at_100"],
        "test_p_at_500": v3_s42["test_p_at_500"],
        "test_p_at_1000": v3_s42["test_p_at_1000"],
        "test_r_at_100": v3_s42["test_r_at_100"],
        "test_r_at_500": v3_s42["test_r_at_500"],
        "test_r_at_1000": v3_s42["test_r_at_1000"],
        "own_val_frozen_tau": v3_s42["own_val_frozen_tau"],
        "test_frozen_precision": v3_s42["test_frozen_precision"],
        "test_frozen_recall": v3_s42["test_frozen_recall"],
        "test_frozen_f1": v3_s42["test_frozen_f1"],
        "test_frozen_fpr": v3_s42["test_frozen_fpr"],
        "test_frozen_tp": v3_s42["test_frozen_tp"],
        "test_frozen_fp": v3_s42["test_frozen_fp"],
        "test_frozen_tn": v3_s42["test_frozen_tn"],
        "test_frozen_fn": v3_s42["test_frozen_fn"],
        "test_050_precision": v3_s42["test_050_precision"],
        "test_050_recall": v3_s42["test_050_recall"],
        "test_050_f1": v3_s42["test_050_f1"],
        "test_050_fpr": v3_s42["test_050_fpr"],
        "test_050_tp": v3_s42["test_050_tp"],
        "test_050_fp": v3_s42["test_050_fp"],
        "metrics_path": v3_s42["metrics_path"],
        "checkpoint_path": v3_s42["checkpoint_path"],
        "val_predictions_path": v3_s42["val_predictions_path"],
        "test_predictions_path": v3_s42["test_predictions_path"],
    })
    comp_rows.append({
        "model_id": "tgat_v3_3seed_mean",
        "model_name": "TGAT v3 (1s, log1p_seconds, 3-seed mean)",
        "model_type": "Temporal GNN (Corrected TimeEncoder)",
        "provenance_status": "Newly Trained (v3, 1s log1p_seconds, seeds 42/43/44)",
        "n_seeds": 3,
        "feature_count": 26,
        "training_seconds": v3_ms["training_seconds"]["mean"],
        "val_roc_auc": v3_ms["val_roc_auc"]["mean"],
        "val_pr_auc": v3_ms["val_pr_auc"]["mean"],
        "test_roc_auc": v3_ms["test_roc_auc"]["mean"],
        "test_pr_auc": v3_ms["test_pr_auc"]["mean"],
        "test_p_at_100": v3_ms["test_p_at_100"]["mean"],
        "test_p_at_500": v3_ms["test_p_at_500"]["mean"],
        "test_p_at_1000": v3_ms["test_p_at_1000"]["mean"],
        "test_r_at_100": v3_ms["test_r_at_100"]["mean"],
        "test_r_at_500": v3_ms["test_r_at_500"]["mean"],
        "test_r_at_1000": v3_ms["test_r_at_1000"]["mean"],
        "own_val_frozen_tau": v3_ms["own_val_frozen_tau"]["mean"],
        "test_frozen_precision": v3_ms["test_frozen_precision"]["mean"],
        "test_frozen_recall": v3_ms["test_frozen_recall"]["mean"],
        "test_frozen_f1": v3_ms["test_frozen_f1"]["mean"],
        "test_frozen_fpr": v3_ms["test_frozen_fpr"]["mean"],
        "test_frozen_tp": v3_ms["test_frozen_tp"]["mean"],
        "test_frozen_fp": v3_ms["test_frozen_fp"]["mean"],
        "test_frozen_tn": 282576.0 - v3_ms["test_frozen_fp"]["mean"],
        "test_frozen_fn": 301.0 - v3_ms["test_frozen_tp"]["mean"],
        "test_050_precision": v3_ms["test_050_precision"]["mean"],
        "test_050_recall": v3_ms["test_050_recall"]["mean"],
        "test_050_f1": v3_ms["test_050_f1"]["mean"],
        "test_050_fpr": v3_ms["test_050_fpr"]["mean"],
        "test_050_tp": v3_ms["test_050_tp"]["mean"],
        "test_050_fp": v3_ms["test_050_fp"]["mean"],
        "metrics_path": str(V3_DIR / "summary_metrics.json"),
        "checkpoint_path": str(V3_DIR / "corrected_tgat_v3_seed42" / "model.pt"),
        "val_predictions_path": str(V3_DIR / "corrected_tgat_v3_seed42" / "validation_predictions.csv"),
        "test_predictions_path": str(V3_DIR / "corrected_tgat_v3_seed42" / "test_predictions.csv"),
    })

    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "baseline_vs_tgat_v3_comparison.csv", index=False)
    (OUT_DIR / "baseline_vs_tgat_v3_comparison.json").write_text(json.dumps(comp_rows, indent=2), encoding="utf-8")

    # 3. Historical v1 (1000s) & v2 Audit
    hist_rows: list[dict[str, Any]] = []
    # v2 runs
    for r in v2_tgat_rows:
        hist_rows.append(r)
    for rid, lbl, folder in [
        ("v2_ablation_no_neighbors", "v2 No Neighbors (1s raw_s)", V2_DIR / "ablation_no_neighbors"),
        ("v2_ablation_no_time_encoding", "v2 No Time Encoding (1s raw_s)", V2_DIR / "ablation_no_time_encoding"),
    ]:
        if (folder / "metrics.json").exists():
            hist_rows.append(extract_tgat_run_record(rid, "Historical v2 Ablations", rid, lbl, folder, "Historical Reference (v2, 1s raw_s)", primary_tau, 1))

    # v1 official & all 15 v1 legacy ablations
    if (MAIN_APP / "results" / "tgat_official_primary" / "metrics.json").exists():
        hist_rows.append(
            extract_tgat_run_record(
                "v1_tgat_official_primary", "Historical v1", "v1_tgat",
                "Legacy TGAT v1 (1000s raw_s)", MAIN_APP / "results" / "tgat_official_primary",
                "Historical Reference (v1, 1000s quantized)", primary_tau, legacy_res=1000
            )
        )
    v1_abl_root = MAIN_APP / "results" / "ablations"
    for mpath in sorted(v1_abl_root.rglob("metrics.json")):
        rdir = mpath.parent
        rel = str(rdir.relative_to(v1_abl_root)).replace("\\", "/")
        mj = json.loads(mpath.read_text(encoding="utf-8"))
        rec_flag = bool(mj.get("evaluation_recovered", False))
        prov = "Recovered Evaluation (v1, 1000s)" if rec_flag else "Historical Reference (v1, 1000s)"
        if rel == "a1_no_temporal_features":
            prov += " [Duplicate of a_no_temporal_features]"
        hist_rows.append(
            extract_tgat_run_record(
                f"v1_abl_{rel.replace('/', '_')}", "Historical v1 Ablations", rel,
                f"Legacy v1 {rel} (1000s)", rdir, prov, primary_tau, legacy_res=1000
            )
        )

    pd.DataFrame(hist_rows).to_csv(OUT_DIR / "historical_v1_and_v2_audit.csv", index=False)
    (OUT_DIR / "historical_v1_and_v2_audit.json").write_text(json.dumps(hist_rows, indent=2), encoding="utf-8")

    # 4. Confusion Matrices Table
    cm_rows = []
    for r in comp_rows:
        cm_rows.append({
            "model_or_run": r["model_name"],
            "provenance": r["provenance_status"],
            "operating_point": f"val_frozen_max_f1 (tau={r['own_val_frozen_tau']:.6f})",
            "precision": r["test_frozen_precision"],
            "recall": r["test_frozen_recall"],
            "f1": r["test_frozen_f1"],
            "fpr": r["test_frozen_fpr"],
            "tp": r["test_frozen_tp"],
            "fp": r["test_frozen_fp"],
            "tn": r["test_frozen_tn"],
            "fn": r["test_frozen_fn"],
        })
    for r in v3_rows:
        if r["group"] == "Full TGAT v3":
            cm_rows.append({
                "model_or_run": f"{r['label']} [@default 0.50]",
                "provenance": r["provenance_status"],
                "operating_point": "default 0.500000",
                "precision": r["test_050_precision"],
                "recall": r["test_050_recall"],
                "f1": r["test_050_f1"],
                "fpr": r["test_050_fpr"],
                "tp": r["test_050_tp"],
                "fp": r["test_050_fp"],
                "tn": r["test_050_tn"],
                "fn": r["test_050_fn"],
            })
            cm_rows.append({
                "model_or_run": f"{r['label']} [@frozen primary tau*]",
                "provenance": r["provenance_status"],
                "operating_point": f"primary_tau*={primary_tau:.6f}",
                "precision": r["test_primary_tau_precision"],
                "recall": r["test_primary_tau_recall"],
                "f1": r["test_primary_tau_f1"],
                "fpr": r["test_frozen_fpr"] if r["run_id"] == "tgat_v3_seed42" else None,
                "tp": r["test_primary_tau_tp"],
                "fp": r["test_primary_tau_fp"],
                "tn": 282576 - int(r["test_primary_tau_fp"]),
                "fn": 301 - int(r["test_primary_tau_tp"]),
            })
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrices_all.csv", index=False)
    (OUT_DIR / "confusion_matrices_all.json").write_text(json.dumps(cm_rows, indent=2), encoding="utf-8")

    # 5. Non-Completed / Scope Boundary Log & Reproducibility Manifest
    non_completed = [
        {
            "experiment": "3-seed replication (seeds 43, 44) for Neighbor-K Sweep (K=1, 5, 10 hist20, 20) and Architecture/Dropout Sweep (e1..e6)",
            "status": "Single Seed (Seed 42) Only in Controlled v3",
            "reason": "Compute budget allocation (~35–42 min per run across 10 configurations × 2 extra seeds ≈ +11–14 GPU-hours). Multi-seed (42, 43, 44) resources were prioritized for Full TGAT v3 and all three core architectural component ablations (No Time Encoding, No Neighbors, No Temporal Behavioural Features).",
        },
        {
            "experiment": "Retraining Static Baselines (Logistic Regression, Static XGBoost, Static GCN) inside v3 directory",
            "status": "Evaluated Directly from Original Preserved Artifacts with Unified Validation-Frozen Thresholding",
            "reason": "These models have zero dependence on 1,000s vs 1s edge timestamp resolution: LR and Static XGBoost consume transaction-split features from data/processed/modeling_splits (identical 5,749,364 train / 891,571 val / 282,877 test rows) and Static GCN consumes TRAIN-only static graph structure. Temporal XGBoost was explicitly retrained under both v1 (1000s) and v2 (1s) regimes.",
        },
        {
            "experiment": "15 Legacy Ablation Folders under c:/Users/duxdh/Downloads/tgat/results/ablations/",
            "status": "Excluded from Controlled v3 Comparisons (Classified Historical v1, 1000s)",
            "reason": "All 15 legacy folders used 1,000-second quantized timestamps and unscaled raw_seconds TimeEncoder; a1_no_temporal_features duplicates a_no_temporal_features; a0_baseline_control and b_no_neighbors have evaluation_recovered=True from checkpoints. All non-duplicate configs were freshly trained from scratch under controlled v3.",
        },
        {
            "experiment": "Future Generalization Window Evaluation on Baselines & Ablations",
            "status": "Executed on Primary TGAT v3 (seed 42), Historical TGAT v2 (seed 42), and App Checkpoint",
            "reason": "Future window parquet splits (223 post-test rows: 134 fraud / 89 non-fraud) serve specifically as the temporal deployment shift stress test for the production TGAT architecture.",
        },
    ]
    (OUT_DIR / "non_completed_experiments_log.json").write_text(json.dumps(non_completed, indent=2), encoding="utf-8")

    manifest_rows = []
    for r in v3_rows:
        manifest_rows.append({
            "id": r["run_id"],
            "category": r["group"],
            "provenance": r["provenance_status"],
            "seed": r["seed"],
            "dataset": r["dataset"],
            "checkpoint": r["checkpoint_path"],
            "config": r["config_path"],
            "scaler": r["scaler_path"],
            "metrics_json": r["metrics_path"],
            "training_log": r["training_log_path"],
            "val_predictions": r["val_predictions_path"],
            "test_predictions": r["test_predictions_path"],
        })
    for r in comp_rows:
        if r["model_id"] not in ("tgat_v2_3seed_mean", "tgat_v3_seed42_primary", "tgat_v3_3seed_mean"):
            manifest_rows.append({
                "id": r["model_id"],
                "category": r["model_type"],
                "provenance": r["provenance_status"],
                "seed": 42,
                "dataset": "modeling_splits / static_gnn / temporal_gnn(_v2_1s)",
                "checkpoint": r["checkpoint_path"],
                "config": "See metrics/code contract",
                "scaler": "N/A or embedded",
                "metrics_json": r["metrics_path"],
                "training_log": "N/A",
                "val_predictions": r["val_predictions_path"],
                "test_predictions": r["test_predictions_path"],
            })
    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "reproducibility_manifest.csv", index=False)
    (OUT_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    # 6. Publication-quality Plots (figA .. figD)
    generate_plots(v3_rows, agg_dict, comp_rows, frozen_th)

    master = {
        "generated_date": date.today().isoformat(),
        "primary_frozen_threshold": frozen_th,
        "v3_multiseed_aggregate": agg_dict.get("full_v3"),
        "controlled_v3_aggregates": agg_dict,
        "controlled_v3_all_runs": v3_rows,
        "baseline_vs_tgat_v3": comp_rows,
        "historical_v1_and_v2_audit": hist_rows,
        "non_completed_or_excluded": non_completed,
    }
    (OUT_DIR / "final_paper_evidence_master.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    print("[EVIDENCE] Built all CSV/JSON tables and plots cleanly in", OUT_DIR)
    return master


def generate_plots(v3_rows: list[dict[str, Any]], agg_dict: dict[str, Any], comp_rows: list[dict[str, Any]], frozen_th: dict[str, Any]) -> None:
    # Fig A: Test ROC & PR curves (Baselines + Full TGAT v3 + V2 + NoNeighbors)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    curve_specs = [
        ("Temporal XGBoost (v2, 1s)", V2_DIR / "temporal_xgboost" / "test_predictions.csv", "#111111", "-", 2.2),
        ("TGAT v3 Seed 42 (log1p_s)", V3_DIR / "corrected_tgat_v3_seed42" / "test_predictions.csv", "#1f77b4", "-", 2.4),
        ("TGAT v3 Seed 43 (log1p_s)", V3_DIR / "corrected_tgat_v3_seed43" / "test_predictions.csv", "#4fa3e0", "--", 1.5),
        ("TGAT v2 Seed 42 (raw_s)", V2_DIR / "corrected_tgat" / "test_predictions.csv", "#d62728", ":", 1.8),
        ("TGAT v3 No Neighbors (s42)", V3_DIR / "ablations" / "b_no_neighbors" / "test_predictions.csv", "#9467bd", "-.", 1.7),
        ("Static XGBoost", MAIN_APP / "results" / "baselines" / "xgboost" / "predictions_test.csv", "#2ca02c", "--", 1.7),
        ("Logistic Regression", MAIN_APP / "results" / "baselines" / "logistic_regression" / "predictions_test.csv", "#ff7f0e", ":", 1.6),
        ("Static GCN", MAIN_APP / "results" / "baselines" / "static_gcn" / "test_predictions.csv", "#8c564b", "-.", 1.5),
    ]
    for label, path, col, ls, lw in curve_specs:
        if not path.exists():
            continue
        yt, st = load_predictions(path)
        roc = roc_auc_score(yt, st)
        pr = average_precision_score(yt, st)
        fpr, tpr, _ = roc_curve(yt, st)
        # downsample curve visually for vector smoothness
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(600, len(fpr)), dtype=int))
        axes[0].plot(fpr[idx], tpr[idx], color=col, linestyle=ls, linewidth=lw, label=f"{label} ({roc:.3f})")

        prec, rec, _ = precision_recall_curve(yt, st)
        idxp = np.unique(np.linspace(0, len(rec) - 1, min(600, len(rec)), dtype=int))
        axes[1].plot(rec[idxp], prec[idxp], color=col, linestyle=ls, linewidth=lw, label=f"{label} ({pr:.4f})")

    axes[0].plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=1.0)
    axes[0].set_title("(a) Test Split ROC Curves (282,877 Txns, 301 Fraud)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=7.6)

    axes[1].axhline(301 / 282877, color="#999999", linestyle="--", linewidth=1.0, label="Base rate (0.00106)")
    axes[1].set_title("(b) Test Split Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 0.28)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=7.6)

    fig.suptitle("Baseline Models vs TGAT v2 and Corrected TGAT v3: Test Ranking Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "figA_roc_and_pr_baselines_vs_tgat_v3.png", bbox_inches="tight")
    plt.close(fig)

    # Fig B: 3-Seed Component Ablation Mean +/- SD Comparison
    comp_fams = ["full_v3", "no_time_encoding", "no_temporal_features", "no_neighbors"]
    comp_names = ["Full TGAT v3\n(Reference)", "No Time\nEncoding", "No Temporal\nFeatures", "No Neighbors\n(K=0)"]
    comp_cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7))
    panels = [
        ("test_pr_auc", "(a) Test PR-AUC (Mean +/- SD, 3 Seeds)", "PR-AUC"),
        ("test_roc_auc", "(b) Test ROC-AUC (Mean +/- SD, 3 Seeds)", "ROC-AUC"),
        ("test_p_at_500", "(c) Test Precision@500 (Mean +/- SD, 3 Seeds)", "Precision@500"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        means = [agg_dict[f][metric]["mean"] for f in comp_fams if f in agg_dict]
        stds = [agg_dict[f][metric]["std"] for f in comp_fams if f in agg_dict]
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=comp_cols[:len(means)], alpha=0.88, width=0.56)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_names[:len(means)], fontsize=8.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if metric == "test_roc_auc":
            ax.set_ylim(0.84, 0.925)
        else:
            ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
        for b, m, sd in zip(bars, means, stds):
            txt = f"{m:.4f}\n±{sd:.4f}" if metric != "test_roc_auc" else f"{m:.3f}\n±{sd:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, m + sd + (0.002 if metric == "test_roc_auc" else max(means) * 0.03), txt, ha="center", va="bottom", fontsize=7.8)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Controlled TGAT v3 Component Ablations Across Seeds 42, 43, and 44 (log1p_seconds, 1s)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "figB_controlled_v3_component_ablations_multiseed.png", bbox_inches="tight")
    plt.close(fig)

    # Fig C: Neighbor Budget K Sweep & Architecture/Dropout Sensitivity (Seed 42)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    k_map = [
        (0, "no_neighbors", "b_no_neighbors"),
        (1, "k_1_hist20", "k_1"),
        (5, "k_5_hist20", "k_5"),
        (10, "k_10_hist20", "k_10"),
        (20, "k_20_hist20", "k_20"),
    ]
    ks = []
    tpr_vals, p500_vals, vpr_vals = [], [], []
    for k_val, fam, _ in k_map:
        matches = [r for r in v3_rows if r["ablation_family"] == fam and r["seed"] == 42]
        if matches:
            ks.append(k_val)
            tpr_vals.append(matches[0]["test_pr_auc"])
            p500_vals.append(matches[0]["test_p_at_500"])
            vpr_vals.append(matches[0]["val_pr_auc"])

    ax = axes[0]
    xpos = np.arange(len(ks))
    ax.plot(xpos, tpr_vals, marker="o", color="#1f77b4", linewidth=2.2, label="Test PR-AUC")
    ax.plot(xpos, [x * 0.25 for x in p500_vals], marker="s", color="#ff7f0e", linestyle="--", linewidth=1.9, label="Test P@500 (scaled x0.25)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"K={k}" + ("\n(No Nbr)" if k == 0 else "") for k in ks], fontsize=8.5)
    ax.set_xlabel("Temporal Neighbor Budget K (max_history=20 for K>=1)")
    ax.set_ylabel("PR-AUC / Scaled Precision@500")
    ax.set_title("(a) Controlled Neighbor Budget Sweep (Seed 42)")
    for xi, tpr, p5 in zip(xpos, tpr_vals, p500_vals):
        ax.annotate(f"PR={tpr:.4f}\nP@500={p5:.3f}", (xi, tpr), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.6)
    ax.set_xlim(-0.55, len(ks) - 0.45)
    ax.set_ylim(0, max(tpr_vals) * 1.32)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    arch_specs = [
        ("Full v3\n(Ref s42)", "tgat_v3_seed42", "#1f77b4"),
        ("1 Layer\n(E1)", "v3_e1_one_layer_s42", "#5c88da"),
        ("1 Head\n(E2)", "v3_e2_one_head_s42", "#17becf"),
        ("Dim 16\n(E3)", "v3_e3_hidden_16_s42", "#9467bd"),
        ("Dim 64\n(E4)", "v3_e4_hidden_64_s42", "#8c564b"),
        ("Drop 0.0\n(E5)", "v3_e5_dropout_0_s42", "#2ca02c"),
        ("Drop 0.4\n(E6)", "v3_e6_dropout_04_s42", "#d62728"),
    ]
    ax2 = axes[1]
    albls, aprs, acols = [], [], []
    by_id = {r["run_id"]: r for r in v3_rows}
    for lbl, rid, col in arch_specs:
        if rid in by_id:
            albls.append(lbl)
            aprs.append(by_id[rid]["test_pr_auc"])
            acols.append(col)
    x2 = np.arange(len(albls))
    bars2 = ax2.bar(x2, aprs, color=acols, alpha=0.88, width=0.58)
    ax2.axhline(by_id["tgat_v3_seed42"]["test_pr_auc"], color="#1f77b4", linestyle=":", linewidth=1.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(albls, fontsize=8.0)
    ax2.set_ylabel("Test PR-AUC")
    ax2.set_title("(b) Controlled Architecture & Regularization Sensitivity (Seed 42)")
    ax2.set_ylim(0, max(aprs) * 1.22)
    for b, val in zip(bars2, aprs):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.0004, f"{val:.4f}", ha="center", va="bottom", fontsize=7.8)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Controlled TGAT v3 Neighborhood Capacity and Architectural Ablations", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "figC_controlled_v3_k_sweep_and_architecture.png", bbox_inches="tight")
    plt.close(fig)

    # Fig D: Frozen-Threshold F1 vs False Positive Trade-off Across Models
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    sel_models = [c for c in comp_rows if c["model_id"] in (
        "logistic_regression", "static_xgboost", "static_gcn",
        "temporal_xgboost_v2_1s", "tgat_v2_3seed_mean", "tgat_v3_seed42_primary", "tgat_v3_3seed_mean"
    )]
    short = {
        "logistic_regression": "LogReg",
        "static_xgboost": "Static XGB",
        "static_gcn": "Static GCN",
        "temporal_xgboost_v2_1s": "Temp XGB v2",
        "tgat_v2_3seed_mean": "TGAT v2 (3s)",
        "tgat_v3_seed42_primary": "TGAT v3 (s42)",
        "tgat_v3_3seed_mean": "TGAT v3 (3s)",
    }
    xs = np.arange(len(sel_models))
    f1_frozen = [m["test_frozen_f1"] for m in sel_models]
    f1_050 = [m["test_050_f1"] for m in sel_models]
    w = 0.36
    b1 = axes[0].bar(xs - w/2, f1_frozen, width=w, color="#1f77b4", label="Val-Frozen max-F1 tau")
    b2 = axes[0].bar(xs + w/2, f1_050, width=w, color="#ff7f0e", label="Default tau = 0.50")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([short[m["model_id"]] for m in sel_models], rotation=18, ha="right")
    axes[0].set_ylabel("Test F1 Score")
    axes[0].set_title("(a) Operating-Point F1: Validation-Frozen tau vs Default 0.50")
    axes[0].set_ylim(0, max(f1_frozen + f1_050) * 1.28)
    for b in b1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.0018, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.0)
    for b in b2:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.0018, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=6.6, color="#b85600")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")

    fprs = [max(m["test_frozen_fpr"] * 100, 1e-4) for m in sel_models]
    recs = [m["test_frozen_recall"] * 100 for m in sel_models]
    fps = [m["test_frozen_fp"] for m in sel_models]
    tps = [m["test_frozen_tp"] for m in sel_models]
    axes[1].scatter(fprs, recs, s=90, c=["#888888", "#2ca02c", "#8c564b", "#111111", "#d62728", "#1f77b4", "#4fa3e0"], zorder=5)
    label_offsets = {
        "logistic_regression": ((-14, 4), "right", "center"),
        "static_xgboost": ((8, 5), "left", "bottom"),
        "static_gcn": ((-8, 8), "right", "bottom"),
        "temporal_xgboost_v2_1s": ((8, 6), "left", "bottom"),
        "tgat_v2_3seed_mean": ((-12, -16), "right", "top"),
        "tgat_v3_seed42_primary": ((8, 6), "left", "bottom"),
        "tgat_v3_3seed_mean": ((8, -10), "left", "top"),
    }
    for m, xpt, ypt, tp, fp in zip(sel_models, fprs, recs, tps, fps):
        off, ha, va = label_offsets.get(m["model_id"], ((6, 4), "left", "bottom"))
        axes[1].annotate(
            f"{short[m['model_id']]}\n(TP={tp:.0f}, FP={fp:.0f})",
            (xpt, ypt),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            va=va,
            fontsize=7.4,
            arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.7} if m["model_id"] in ("logistic_regression", "tgat_v2_3seed_mean", "tgat_v3_3seed_mean") else None,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.06, 12.0)
    axes[1].set_ylim(2.0, 32.5)
    axes[1].set_xlabel("Test False Positive Rate (%) [Log Scale]")
    axes[1].set_ylabel("Test Fraud Recall (%)")
    axes[1].set_title("(b) Operational Alert Burden at Frozen Thresholds")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Validation-Frozen Decision Threshold Calibration Across Models (Test Split)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "figD_threshold_and_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    if args.plots_only:
        master = json.loads((OUT_DIR / "final_paper_evidence_master.json").read_text(encoding="utf-8"))
        generate_plots(
            master["controlled_v3_all_runs"],
            master["controlled_v3_aggregates"],
            master["baseline_vs_tgat_v3"],
            master["primary_frozen_threshold"],
        )
        print("[EVIDENCE] Regenerated plots cleanly in", PLOTS_DIR)
    else:
        build_all()

