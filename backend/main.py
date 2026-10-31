"""FastAPI Application for Explainable TGAT Financial Fraud Ring Detection.

Serves real-time TGAT inference, dynamic graph attention explainability,
batch transaction upload analysis, interactive graph queries, and evaluation benchmarks.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.analysis_store import analysis_store
from backend.llm_explainer import answer_account_question, explain_account_network, explain_batch_overview
from backend.model_service import ModelService
from backend.validation import ValidationResult, detect_and_validate_csv

app = FastAPI(
    title="Explainable TGAT Financial Fraud Ring Detection API",
    description="Backend service providing real-time TGAT inference, dynamic graph attention explainability, batch analysis, and evaluation benchmarks.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Ensure static files and API responses are never stale-cached by client browsers."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

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


class BatchRunRequest(BaseModel):
    filename: str = "upload.csv"
    csv_content: str


class AccountQuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    # Preload model on startup
    get_model_service()


# ============================================================================
# SYSTEM HEALTH & BENCHMARKS
# ============================================================================

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
        df = pd.read_csv(sweep_csv)
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


# ============================================================================
# NEW WORKFLOW: UPLOAD, VALIDATE, ANALYSE, RESULTS & GRAPH
# ============================================================================

@app.post("/api/analysis/upload")
async def upload_csv_file(request: Request):
    """Upload and validate a transaction CSV file."""
    content_type = request.headers.get("content-type", "")
    filename = request.headers.get("x-filename", "upload.csv")
    body = await request.body()
    if "application/json" in content_type:
        try:
            data = json.loads(body.decode("utf-8"))
            content = data.get("csv_content", "").encode("utf-8")
            filename = data.get("filename", filename)
        except Exception:
            content = body
    else:
        content = body

    val_result, _ = detect_and_validate_csv(content, filename=filename)
    return {
        "filename": filename,
        "size_bytes": len(content),
        "validation": val_result,
    }


@app.post("/api/analysis/validate")
def validate_csv_content(payload: BatchRunRequest):
    """Validate raw CSV text content and return diagnostics."""
    content = payload.csv_content.encode("utf-8")
    val_result, _ = detect_and_validate_csv(content, filename=payload.filename)
    return val_result


@app.post("/api/analysis/run")
def run_batch_analysis(payload: BatchRunRequest):
    """Execute causal batch TGAT scoring on validated CSV content."""
    ms = get_model_service()
    content = payload.csv_content.encode("utf-8")
    val_result, valid_df = detect_and_validate_csv(content, filename=payload.filename)

    if not val_result.valid or valid_df is None or len(valid_df) == 0:
        detail_msg = val_result.errors[0].message if val_result.errors else "Uploaded file failed schema validation. Correct errors before running."
        raise HTTPException(status_code=400, detail=detail_msg)

    t0 = time.time()
    scored_records = ms.predict_batch(valid_df, causal=True)
    duration = time.time() - t0

    session = analysis_store.add_session(
        filename=payload.filename,
        transactions=scored_records,
        threshold=ms.threshold,
        warnings=val_result.warnings,
        duration_seconds=duration,
    )

    return {
        "status": "success",
        "session_id": session.session_id,
        "filename": session.filename,
        "total_transactions": session.total_transactions,
        "flagged_transactions": session.flagged_transactions,
        "flagged_rate": session.flagged_rate,
        "distinct_accounts": session.distinct_accounts,
        "threshold": session.threshold,
        "duration_seconds": session.duration_seconds,
        "warnings": session.warnings,
    }


@app.get("/api/analysis/sample")
def get_sample_benchmark_data(count: int = Query(50, ge=10, le=2000)):
    """Return a real sample from test.csv for instant 5-minute academic demonstration."""
    ms = get_model_service()
    try:
        sample_df = ms.get_benchmark_sample(count=count)
        csv_str = sample_df.to_csv(index=False)
        return {
            "filename": f"ibm_aml_benchmark_sample_{count}tx.csv",
            "row_count": len(sample_df),
            "csv_content": csv_str,
            "preview": sample_df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample: {str(e)}")


@app.get("/api/analysis/overview")
def get_analysis_overview(session_id: Optional[str] = Query(None)):
    """Return summary statistics, histogram distribution, and top flagged alerts for selected analysis."""
    session = analysis_store.get_session(session_id)
    if not session:
        return {
            "has_data": False,
            "message": "No transaction analyses run yet. Upload a transaction CSV to begin.",
        }

    top_flagged = [t for t in session.transactions if t["classification"] == "FLAGGED"][:5]

    overview_payload = {
        "session_id": session.session_id,
        "total_transactions": session.total_transactions,
        "flagged_transactions": session.flagged_transactions,
        "high_risk_accounts": session.high_risk_accounts,
        "threshold": session.threshold,
    }
    briefing_res = explain_batch_overview(overview_payload)

    return {
        "has_data": True,
        "session_id": session.session_id,
        "filename": session.filename,
        "created_at": session.created_at,
        "total_transactions": session.total_transactions,
        "flagged_transactions": session.flagged_transactions,
        "flagged_rate": session.flagged_rate,
        "high_risk_count": session.high_risk_count,
        "medium_risk_count": session.medium_risk_count,
        "low_risk_count": session.low_risk_count,
        "high_risk_accounts": session.high_risk_accounts,
        "executive_summary": session.executive_summary,
        "ai_briefing": briefing_res.get("briefing", ""),
        "ai_briefing_model": briefing_res.get("model_used", "AI Explainer"),
        "distinct_accounts": session.distinct_accounts,
        "threshold": session.threshold,
        "duration_seconds": session.duration_seconds,
        "score_distribution": session.score_distribution,
        "warnings": session.warnings,
        "recent_flagged": top_flagged,
    }


@app.get("/api/analysis/sessions")
def list_analysis_sessions():
    """List all completed analysis sessions."""
    return {"sessions": analysis_store.list_sessions()}


@app.get("/api/analysis/{session_id}/results")
def get_analysis_results(
    session_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=5, le=100),
    sort_by: str = Query("risk_score"),
    sort_dir: str = Query("desc"),
    search: Optional[str] = Query(None),
    flag_filter: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None),
):
    """Return server-side paginated, sorted, searchable results."""
    return analysis_store.get_results(
        session_id=session_id,
        page=page,
        limit=limit,
        sort_by=sort_by,
        sort_dir=sort_dir,
        search=search,
        flag_filter=flag_filter,
        risk_level=risk_level,
        min_score=min_score,
        max_score=max_score,
    )


@app.get("/api/analysis/{session_id}/graph")
def get_analysis_graph(
    session_id: str,
    min_score: Optional[float] = Query(None),
    account: Optional[str] = Query(None),
):
    """Return connected accounts directed graph nodes and edges."""
    return analysis_store.get_graph(
        session_id=session_id,
        min_score=min_score,
        account_filter=account,
    )


@app.get("/api/analysis/{session_id}/transaction/{tx_id}")
def get_transaction_detail(session_id: str, tx_id: str):
    """Return individual transaction details including layer-wise attention distribution and factual evidence."""
    tx = analysis_store.get_transaction(session_id, tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return tx


@app.get("/api/analysis/{session_id}/account/{account_id}")
def get_account_detail(session_id: str, account_id: str):
    """Return account history, counterparty interaction volume, and LLM network explanation."""
    acc = analysis_store.get_account_history(session_id, account_id)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found in analysis session.")

    session = analysis_store.get_session(session_id)
    thresh = session.threshold if session else 0.737155
    explanation = explain_account_network(acc, session_threshold=thresh)
    acc["ai_narrative"] = explanation.get("narrative", "")
    acc["ai_model_used"] = explanation.get("model_used", "AI Explainer")
    acc["ai_network_context"] = explanation.get("network_context", {})
    return acc


@app.post("/api/analysis/{session_id}/account/{account_id}/ask")
def ask_account_question_endpoint(session_id: str, account_id: str, payload: AccountQuestionRequest):
    """Ask an interactive follow-up question about this account and its network topology."""
    acc = analysis_store.get_account_history(session_id, account_id)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found in analysis session.")

    session = analysis_store.get_session(session_id)
    thresh = session.threshold if session else 0.737155
    return answer_account_question(acc, payload.question, session_threshold=thresh)


@app.get("/api/analysis/{session_id}/export")
def export_analysis_csv(session_id: str):
    """Download genuine scored analysis results as CSV."""
    csv_str = analysis_store.export_csv(session_id)
    if not csv_str:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=tgat_analysis_{session_id}.csv"},
    )


# Mount Static Frontend
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
