"""FastAPI Application for Explainable TGAT Financial Fraud Ring Detection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.model_service import ModelService

app = FastAPI(
    title="Explainable TGAT Financial Fraud Ring Detection API",
    description="Backend service providing real-time TGAT inference, dynamic graph attention explainability, and evaluation benchmarks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


# Request Models
class TransactionInput(BaseModel):
    from_account: str
    to_account: str
    timestamp: Optional[int] = 1662800000
    amount: float = 1000.0
    amount_paid: Optional[float] = None
    same_bank: bool = False
    cross_bank: bool = True
    is_weekend: bool = False
    same_currency: bool = True
    custom_features: Optional[Dict[str, float]] = None


@app.on_event("startup")
def startup_event():
    # Preload model on startup
    get_model_service()


@app.get("/api/health")
def health():
    ms = get_model_service()
    return {
        "status": "healthy",
        "service": "Explainable TGAT Fraud Detection Engine",
        "model_architecture": "Temporal Graph Attention Network (TGAT)",
        "device": str(ms.device),
        "feature_dim": len(ms.features),
        "frozen_threshold": ms.threshold,
        "checkpoint_file": "results/tgat/model.pt",
    }


@app.get("/api/metrics")
def get_metrics():
    """Return verified metrics for TGAT and all baseline models."""
    metrics_path = Path("results/threshold_analysis/tgat/test_at_frozen_threshold.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        tgat_test_frozen = json.load(f)

    with open("results/tgat/metrics.json", "r", encoding="utf-8") as f:
        tgat_raw = json.load(f)

    # Baselines
    baselines = {}
    baseline_dir = Path("results/baselines")
    if baseline_dir.exists():
        for b in ["logistic_regression", "static_gcn", "xgboost", "temporal_xgboost"]:
            b_path = baseline_dir / b / "metrics.json"
            if b_path.exists():
                with open(b_path, "r", encoding="utf-8") as f:
                    bm = json.load(f)
                    val = bm.get("metrics", {}).get("validation") or bm.get("validation", {})
                    test = bm.get("metrics", {}).get("test") or bm.get("test", {})
                    baselines[b] = {"validation": val, "test": test}

    return {
        "tgat_frozen_test": tgat_test_frozen,
        "tgat_official_metrics": tgat_raw,
        "baselines": baselines,
        "summary": {
            "total_transactions": 6924035,
            "train_rows": 5749364,
            "val_rows": 891571,
            "test_rows": 282877,
            "future_rows": 223,
            "test_positives": 301,
            "frozen_threshold": 0.737155,
            "precision_gain": "+138% (1.91% -> 4.55%)",
            "false_positive_reduction": "-83.8% (2,198 -> 357)",
        }
    }


@app.get("/api/threshold")
def get_threshold_details():
    """Return threshold optimization sweep and curve diagnostic points."""
    with open("results/threshold_analysis/frozen_threshold.json", "r", encoding="utf-8") as f:
        frozen = json.load(f)

    sweep_csv = Path("results/threshold_analysis/tgat/validation_sweep.csv")
    sweep_data = []
    if sweep_csv.exists():
        import pandas as pd
        df = pd.read_csv(sweep_csv)
        # Sample 50 points evenly for fast client rendering
        step = max(1, len(df) // 50)
        sampled = df.iloc[::step]
        sweep_data = sampled.to_dict(orient="records")

    return {
        "frozen": frozen,
        "sweep_sample": sweep_data,
        "operational_tradeoff": {
            "default_threshold": 0.50,
            "optimal_threshold": 0.737155,
            "fp_reduction_percent": 83.76,
            "precision_boost_percent": 138.2,
        }
    }


@app.get("/api/ablations")
def get_ablations():
    """Return all 15 ablation experiments with granular metric attributions."""
    abl_dir = Path("results/ablations")
    runs = []
    if abl_dir.exists():
        for root, dirs, files in os.walk(abl_dir):
            if "metrics.json" in files:
                rel = os.path.relpath(root, abl_dir)
                with open(os.path.join(root, "metrics.json"), "r", encoding="utf-8") as f:
                    m = json.load(f)
                cfg = {}
                if "config.json" in files:
                    with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                val = m.get("validation", {})
                test = m.get("test", {})
                runs.append({
                    "id": rel.replace("\\", "/"),
                    "name": rel.replace("\\", "/"),
                    "epochs": cfg.get("epochs", 3),
                    "neighbor_k": cfg.get("neighbor_k", 10),
                    "ablation_type": cfg.get("ablation", "none"),
                    "num_layers": cfg.get("num_layers", 2),
                    "num_heads": cfg.get("num_heads", 2),
                    "hidden_dim": cfg.get("hidden_dim", 32),
                    "dropout": cfg.get("dropout", 0.2),
                    "val_roc_auc": val.get("roc_auc"),
                    "val_pr_auc": val.get("pr_auc"),
                    "test_roc_auc": test.get("roc_auc"),
                    "test_pr_auc": test.get("pr_auc"),
                    "test_f1": test.get("f1"),
                    "test_p100": test.get("precision_at_100"),
                    "test_r100": test.get("recall_at_100"),
                    "training_seconds": m.get("training_seconds"),
                })
    return {"count": len(runs), "experiments": sorted(runs, key=lambda x: x["id"])}


@app.get("/api/cases")
def get_cases():
    """Return curated fraud ring case studies with graph topology."""
    cases_path = Path("backend/cases_data.json")
    if not cases_path.exists():
        raise HTTPException(status_code=404, detail="Case studies not generated yet.")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {"cases": cases}


@app.get("/api/cases/{case_id}")
def get_case_by_id(case_id: str):
    cases_path = Path("backend/cases_data.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        if c["id"] == case_id:
            return c
    raise HTTPException(status_code=404, detail="Case study not found.")


@app.post("/api/predict")
def predict_transaction(input_data: TransactionInput):
    """Run live TGAT inference and return prediction with dynamic attention weights."""
    ms = get_model_service()

    amount_received = float(input_data.amount)
    amount_paid = float(input_data.amount_paid if input_data.amount_paid is not None else amount_received)
    amount_diff = abs(amount_received - amount_paid)
    amount_ratio = amount_received / amount_paid if amount_paid > 0 else 1.0

    features = {
        "Amount Received": amount_received,
        "Amount Paid": amount_paid,
        "Amount Difference": amount_diff,
        "Amount Ratio": amount_ratio,
        "Same Bank Transaction": 1.0 if input_data.same_bank else 0.0,
        "Cross Bank Transaction": 1.0 if input_data.cross_bank else 0.0,
        "Transaction Time Category": 2.0,  # Afternoon
        "Is Weekend": 1.0 if input_data.is_weekend else 0.0,
        "Log Amount Received": float(np.log1p(amount_received)),
        "Log Amount Paid": float(np.log1p(amount_paid)),
        "Same Currency": 1.0 if input_data.same_currency else 0.0,
        "sender_in_count": 4.0,
        "sender_out_count": 8.0,
        "sender_total_count": 12.0,
        "sender_in_amount": amount_received * 2.5,
        "sender_out_amount": amount_paid * 3.8,
        "sender_avg_in_amount": amount_received * 0.6,
        "sender_avg_out_amount": amount_paid * 0.45,
        "sender_time_since_last": 120.0,
        "receiver_in_count": 2.0,
        "receiver_out_count": 1.0,
        "receiver_total_count": 3.0,
        "receiver_in_amount": amount_received * 1.1,
        "receiver_out_amount": amount_paid * 0.5,
        "receiver_avg_in_amount": amount_received * 0.55,
        "receiver_time_since_last": 3600.0,
    }

    if input_data.custom_features:
        features.update(input_data.custom_features)

    result = ms.predict(
        from_account=input_data.from_account,
        to_account=input_data.to_account,
        timestamp_sec=input_data.timestamp or 1662800000,
        feature_dict=features,
    )
    return result


# Mount Static Frontend
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
