"""Recover evaluation artifacts for an already-trained TGAT checkpoint.

This script does not train or update model weights. It reconstructs the
train-only scaler and chronological history, loads the saved checkpoint, and
writes validation/test metrics and prediction CSVs into the configured results
directory. Existing evaluation artifacts are never overwritten.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.tgat_dataset import ALL_FEATURES, EventFeatureStore, TARGET
from src.models.temporal.train_tgat import evaluate, populate_history, predict, read_frame


RESULTS_DIR = Path("results/ablations/a0_baseline_control")
OUTPUT_NAMES = (
    "metrics.json",
    "validation_predictions.csv",
    "test_predictions.csv",
)


def main() -> None:
    config_path = RESULTS_DIR / "config.json"
    checkpoint_path = RESULTS_DIR / "model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    existing = [name for name in OUTPUT_NAMES if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing recovery outputs: "
            + ", ".join(existing)
            + ". Move or rename them first if you intentionally want to rerun recovery."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = Path(config["data_dir"])
    active_features = config["active_features"]
    device = torch.device("cpu")
    evaluation_started = time.time()

    print("Loading chronological train/validation/test splits...", flush=True)
    train = read_frame(data_dir / "train.csv", config.get("max_train_rows"))
    validation = read_frame(
        data_dir / "validation.csv", config.get("max_validation_rows")
    )
    test = read_frame(data_dir / "test.csv", config.get("max_test_rows"))

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise RuntimeError("One or more TGAT input splits are empty.")
    if list(train[ALL_FEATURES].columns) != ALL_FEATURES:
        raise RuntimeError("TGAT feature contract mismatch.")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("Training split contains invalid target labels.")

    print("Fitting the train-only scaler and transforming each split once...", flush=True)
    scaler = TemporalFeatureScaler.fit(train)
    train = scaler.transform_frame(train)
    validation = scaler.transform_frame(validation)
    test = scaler.transform_frame(test)

    model = TGATTransactionModel(
        transaction_dim=config["feature_count"],
        event_dim=config["feature_count"],
        hidden_dim=config["hidden_dim"],
        time_dim=16,
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        use_time_encoding=config["use_time_encoding"],
    ).to(device)

    print("Loading the saved checkpoint (no training will run)...", flush=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("Rebuilding chronological training history...", flush=True)
    store = EventFeatureStore(max_history=config["max_history"])
    history_batches = populate_history(
        train, store, config["batch_size"], active_features
    )
    print(
        f"History rebuilt: {len(train):,} rows across {history_batches:,} batches.",
        flush=True,
    )

    print("Evaluating validation split...", flush=True)
    val_labels, val_scores, val_timestamps = predict(
        model,
        validation,
        store,
        config["batch_size"],
        config["neighbor_k"],
        device,
        active_features,
        config["use_neighbors"],
    )

    # predict() inserts validation events into the store. Therefore test
    # prediction correctly continues from train + validation history.
    print("Evaluating test split...", flush=True)
    test_labels, test_scores, test_timestamps = predict(
        model,
        test,
        store,
        config["batch_size"],
        config["neighbor_k"],
        device,
        active_features,
        config["use_neighbors"],
    )

    metrics = {
        "validation": evaluate(val_labels, val_scores),
        "test": evaluate(test_labels, test_scores),
        "evaluation_seconds": time.time() - evaluation_started,
        "feature_count": config["feature_count"],
        "neighbor_k": config["neighbor_k"],
        "max_history": config["max_history"],
        "ablation": config["ablation"],
        "negative_sampling_ratio": config["negative_ratio"],
        "device": str(device),
        "feature_scaling": config["feature_scaling"],
        "checkpoint_source": str(checkpoint_path),
        "checkpoint_retrained": False,
        "evaluation_recovered": True,
        "history_batches": history_batches,
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }

    # Write temporary files first, then rename them into place. The upfront
    # existence guard above avoids replacing any existing evaluation artifacts.
    outputs = {
        "metrics.json": json.dumps(metrics, indent=2, default=float),
    }
    prediction_frames = {
        "validation_predictions.csv": pd.DataFrame(
            {"timestamp": val_timestamps, "label": val_labels, "score": val_scores}
        ),
        "test_predictions.csv": pd.DataFrame(
            {"timestamp": test_timestamps, "label": test_labels, "score": test_scores}
        ),
    }

    temp_paths: list[Path] = []
    try:
        for name, text in outputs.items():
            temp = RESULTS_DIR / (name + ".recovering")
            temp.write_text(text, encoding="utf-8")
            temp_paths.append(temp)
        for name, frame in prediction_frames.items():
            temp = RESULTS_DIR / (name + ".recovering")
            frame.to_csv(temp, index=False)
            temp_paths.append(temp)

        for name in OUTPUT_NAMES:
            temp = RESULTS_DIR / (name + ".recovering")
            temp.replace(RESULTS_DIR / name)
            temp_paths.remove(temp)
    finally:
        for temp in temp_paths:
            if temp.exists():
                temp.unlink()

    print("\nTGAT EVALUATION RECOVERY: COMPLETED", flush=True)
    print(json.dumps(metrics, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
