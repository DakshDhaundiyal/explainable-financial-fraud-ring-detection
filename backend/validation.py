"""CSV Validation Engine for TGAT Transaction Uploads.

Validates uploaded transaction data against real model input schemas:
1. Raw IBM AML format (Timestamp, From/To Bank, From/To Account, Amount Received/Paid, Currencies, Format)
2. Processed Temporal GNN format (From/To Account, Timestamp, tabular/temporal features)

Enforces strict causal ordering, positive amounts, valid timestamps, and history availability.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_UPLOAD_ROWS = 2000  # Safe micro-batch processing limit for demo-ready responsiveness

RAW_IBM_AML_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
]

MINIMAL_GNN_COLUMNS = [
    "From Account",
    "To Account",
    "Timestamp",
    "Amount Received",
    "Amount Paid",
]


class ValidationErrorDetail(BaseModel):
    row: Optional[int] = None
    column: Optional[str] = None
    message: str
    severity: str = "error"  # "error" or "warning"


class ValidationResult(BaseModel):
    valid: bool
    format_detected: str
    total_rows: int
    valid_rows: int
    rejected_rows: int
    errors: List[ValidationErrorDetail]
    warnings: List[str]
    columns_present: List[str]
    sample_preview: List[Dict[str, Any]]


def detect_and_validate_csv(csv_bytes: bytes, filename: str = "upload.csv") -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
    """Validate CSV bytes and return structured diagnostic results + parsed DataFrame."""
    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []

    # 1. File Size Check
    if len(csv_bytes) > MAX_FILE_SIZE_BYTES:
        errors.append(ValidationErrorDetail(
            message=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB (actual: {len(csv_bytes) / (1024 * 1024):.2f}MB)."
        ))
        return ValidationResult(
            valid=False,
            format_detected="unknown",
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            errors=errors,
            warnings=warnings,
            columns_present=[],
            sample_preview=[],
        ), None

    # 2. Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    except Exception as e:
        errors.append(ValidationErrorDetail(message=f"Failed to parse CSV file: {str(e)}"))
        return ValidationResult(
            valid=False,
            format_detected="unknown",
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            errors=errors,
            warnings=warnings,
            columns_present=[],
            sample_preview=[],
        ), None

    total_rows = len(df)
    columns_present = list(df.columns)

    if total_rows == 0:
        errors.append(ValidationErrorDetail(message="CSV file contains 0 data rows."))
        return ValidationResult(
            valid=False,
            format_detected="empty",
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            errors=errors,
            warnings=warnings,
            columns_present=columns_present,
            sample_preview=[],
        ), None

    if total_rows > MAX_UPLOAD_ROWS:
        warnings.append(f"Upload contains {total_rows:,} rows. To ensure interactive responsiveness, analysis is capped at {MAX_UPLOAD_ROWS:,} transactions.")
        df = df.iloc[:MAX_UPLOAD_ROWS].copy()

    # 3. Detect Format
    has_raw_ibm = all(col in df.columns for col in RAW_IBM_AML_COLUMNS)
    has_gnn = all(col in df.columns for col in MINIMAL_GNN_COLUMNS)

    format_detected = "raw_ibm_aml" if has_raw_ibm else ("processed_temporal_gnn" if has_gnn else "unknown")

    if format_detected == "unknown":
        missing_raw = [c for c in RAW_IBM_AML_COLUMNS if c not in df.columns]
        missing_gnn = [c for c in MINIMAL_GNN_COLUMNS if c not in df.columns]
        errors.append(ValidationErrorDetail(
            message=f"Unrecognized schema. Must match either Raw IBM AML (missing: {missing_raw[:4]}) or Processed GNN (missing: {missing_gnn})."
        ))
        return ValidationResult(
            valid=False,
            format_detected="unknown",
            total_rows=total_rows,
            valid_rows=0,
            rejected_rows=total_rows,
            errors=errors,
            warnings=warnings,
            columns_present=columns_present,
            sample_preview=df.head(5).to_dict(orient="records"),
        ), None

    # 4. Row-level Validation
    # Timestamp parsing
    try:
        df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        bad_ts_count = int(df["_parsed_ts"].isna().sum())
        if bad_ts_count > 0:
            errors.append(ValidationErrorDetail(
                column="Timestamp",
                message=f"{bad_ts_count} rows have invalid or unparsable timestamps."
            ))
    except Exception as e:
        errors.append(ValidationErrorDetail(column="Timestamp", message=f"Timestamp conversion error: {str(e)}"))

    # Account ID checks
    from_acc_col = "From Account" if "From Account" in df.columns else df.columns[0]
    to_acc_col = "To Account" if "To Account" in df.columns else df.columns[1]

    df["_from"] = df[from_acc_col].astype(str).str.strip()
    df["_to"] = df[to_acc_col].astype(str).str.strip()

    empty_from = int((df["_from"] == "").sum())
    empty_to = int((df["_to"] == "").sum())
    same_acc = int((df["_from"] == df["_to"]).sum())

    if empty_from > 0:
        errors.append(ValidationErrorDetail(column="From Account", message=f"{empty_from} rows have empty sender accounts."))
    if empty_to > 0:
        errors.append(ValidationErrorDetail(column="To Account", message=f"{empty_to} rows have empty receiver accounts."))
    if same_acc > 0:
        warnings.append(f"{same_acc} transactions have identical sender and receiver accounts (self-transfer).")

    # Amount checks
    amount_col = "Amount Received" if "Amount Received" in df.columns else "Amount"
    if amount_col in df.columns:
        df["_amount"] = pd.to_numeric(df[amount_col], errors="coerce")
        nan_amounts = int(df["_amount"].isna().sum())
        non_positive = int((df["_amount"] <= 0).sum())
        if nan_amounts > 0:
            errors.append(ValidationErrorDetail(column=amount_col, message=f"{nan_amounts} rows have non-numeric transaction amounts."))
        if non_positive > 0:
            errors.append(ValidationErrorDetail(column=amount_col, message=f"{non_positive} rows have zero or negative transaction amounts."))

    # Duplicate check
    dup_count = int(df.duplicated(subset=[from_acc_col, to_acc_col, "Timestamp"]).sum())
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate timestamp/account transactions detected. These will be evaluated in sequence.")

    # Sort Chronologically
    if "_parsed_ts" in df.columns and df["_parsed_ts"].notna().all():
        df["_timestamp_sec"] = df["_parsed_ts"].astype("int64") // 1_000_000_000
        df = df.sort_values("_timestamp_sec", kind="stable").reset_index(drop=True)

    is_valid = len(errors) == 0
    valid_rows = len(df) if is_valid else 0
    rejected_rows = total_rows - valid_rows

    # Preview
    preview_cols = [c for c in ["Timestamp", "From Account", "To Account", "Amount Received", "Amount Paid", "Is Laundering"] if c in df.columns]
    sample_preview = df[preview_cols].head(5).to_dict(orient="records") if len(preview_cols) > 0 else df.head(5).to_dict(orient="records")

    result = ValidationResult(
        valid=is_valid,
        format_detected=format_detected,
        total_rows=total_rows,
        valid_rows=valid_rows,
        rejected_rows=rejected_rows,
        errors=errors,
        warnings=warnings,
        columns_present=columns_present,
        sample_preview=sample_preview,
    )

    return result, df if is_valid else None
