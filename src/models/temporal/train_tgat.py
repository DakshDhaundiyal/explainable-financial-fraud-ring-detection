"""Train and evaluate the leakage-safe TGAT transaction classifier.

The stream is chronological. For every timestamp group, neighborhoods are
constructed from the state before that timestamp; only after the examples are
created are those transactions inserted into temporal history. Validation and
test continue the history from earlier splits and never update model weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.models.common.evaluation import classification_metrics
from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import TGATTransactionModel, VALID_TIME_ENCODING_MODES
from src.models.temporal.tgat_dataset import (
    ALL_FEATURES,
    CURRENT_FEATURES,
    TARGET,
    EventFeatureStore,
    build_batch,
    build_batch_from_arrays,
    to_unix_seconds,
)

DEFAULT_DATA = Path("data/processed/temporal_gnn")
DEFAULT_RESULTS = Path("results/tgat")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_frame(path: Path, rows: int | None = None, tail_rows: int | None = None) -> pd.DataFrame:
    if tail_rows is not None and rows is not None:
        raise ValueError("Specify at most one of rows or tail_rows")
    if tail_rows is not None:
        with open(path, "rb") as f:
            total_lines = sum(1 for _ in f)
        data_rows = max(0, total_lines - 1)
        skip_count = max(0, data_rows - int(tail_rows))
        skiprows = range(1, skip_count + 1) if skip_count > 0 else None
        frame = pd.read_csv(path, skiprows=skiprows)
    else:
        frame = pd.read_csv(path, nrows=rows)
    parsed = pd.to_datetime(frame["Timestamp"], format="mixed", errors="raise")
    frame["_timestamp"] = to_unix_seconds(parsed)
    frame = frame.sort_values("_timestamp", kind="stable").reset_index(drop=True)
    return frame


def _row_layout(frame: pd.DataFrame, active_features=None) -> tuple[int, int, int, tuple[int, ...]]:
    """Cache the positional layout used by chronological row processing."""
    columns = list(frame.columns)
    idx = {name: pos for pos, name in enumerate(columns)}
    return (
        idx["From Account"],
        idx["To Account"],
        idx["_timestamp"],
        tuple(idx[name] for name in (active_features or ALL_FEATURES)),
    )


def _group_boundaries(timestamps_arr: np.ndarray) -> list[int]:
    n_rows = len(timestamps_arr)
    if n_rows == 0:
        return []
    splits = np.flatnonzero(timestamps_arr[1:] != timestamps_arr[:-1]) + 1
    boundaries = splits.tolist()
    boundaries.append(n_rows)
    return boundaries


def _resolve_account_ids(accounts: list[str], store: EventFeatureStore) -> list[int]:
    acct_map = store.account_to_id
    next_id = len(acct_map)
    resolved = [0] * len(accounts)
    for idx, acc in enumerate(accounts):
        aid = acct_map.get(acc)
        if aid is None:
            aid = next_id
            acct_map[acc] = aid
            next_id += 1
        resolved[idx] = aid
    return resolved


def _add_slice_to_store(
    senders: np.ndarray,
    receivers: np.ndarray,
    timestamps_arr: np.ndarray,
    feature_matrix: np.ndarray,
    start: int,
    end: int,
    store: EventFeatureStore,
) -> None:
    """Insert a chronological slice into history without constructing pandas objects."""
    add_event = store.store.add_event
    get_id = store.get_id
    for idx in range(start, end):
        src_id = get_id(str(senders[idx]))
        dst_id = get_id(str(receivers[idx]))
        add_event(
            int(timestamps_arr[idx]),
            src_id,
            dst_id,
            feature_matrix[idx],
        )


def _add_resolved_slice_to_store(
    sender_ids: list[int],
    receiver_ids: list[int],
    ts_list: list[int],
    feature_matrix: np.ndarray,
    start: int,
    end: int,
    store: EventFeatureStore,
) -> None:
    """Insert a chronological slice using pre-resolved node IDs and read-only views."""
    add_event = store.store.add_event
    for idx in range(start, end):
        add_event(
            ts_list[idx],
            sender_ids[idx],
            receiver_ids[idx],
            feature_matrix[idx],
        )


def _add_chunk_to_store(
    chunk: pd.DataFrame,
    store: EventFeatureStore,
    source_idx: int,
    destination_idx: int,
    timestamp_idx: int,
    feature_indices: tuple[int, ...],
) -> None:
    """Insert a chronological chunk into history without model inference."""
    store_feature_indices = (source_idx, destination_idx, *feature_indices)
    for row in chunk.itertuples(index=False, name=None):
        store.add_row(
            row,
            int(row[timestamp_idx]),
            feature_indices=store_feature_indices,
        )


def _extract_split_arrays(frame: pd.DataFrame, active_features=None, resolve_ids: bool = True, store: EventFeatureStore | None = None):
    active = list(active_features or ALL_FEATURES)
    timestamps_arr = frame["_timestamp"].to_numpy(dtype=np.int64)
    ts_list = timestamps_arr.tolist()
    group_ends = _group_boundaries(timestamps_arr)
    feature_matrix = np.ascontiguousarray(frame[active].to_numpy(dtype=np.float32))
    labels_arr = np.ascontiguousarray(frame[TARGET].to_numpy(dtype=np.float32))
    if resolve_ids and store is not None:
        sender_ids = _resolve_account_ids(frame["From Account"].astype(str).tolist(), store)
        receiver_ids = _resolve_account_ids(frame["To Account"].astype(str).tolist(), store)
    else:
        sender_ids = None
        receiver_ids = None
    return timestamps_arr, ts_list, group_ends, feature_matrix, labels_arr, sender_ids, receiver_ids


def make_batches(frame: pd.DataFrame, store: EventFeatureStore, batch_size: int, neighbor_k: int, active_features=None, use_neighbors=True):
    """Yield causal micro-batches without exposing same-timestamp events.

    Timestamp groups are the causal state boundary, while batch_size is the
    computational batch size. Every micro-batch from timestamp T is built
    against the same pre-T history. The complete timestamp group is inserted
    into history only after its final micro-batch has been yielded.
    """
    timestamps_arr, ts_list, group_ends, feature_matrix, labels_arr, sender_ids, receiver_ids = _extract_split_arrays(
        frame, active_features, resolve_ids=use_neighbors, store=store
    )

    start = 0
    for group_end in group_ends:
        # Frames are already transformed by the train-only scaler. Passing the
        # scaler again here would double-transform query features while stored
        # event features remain single-transformed.
        for micro_start in range(start, group_end, batch_size):
            micro_end = min(micro_start + batch_size, group_end)
            yield build_batch_from_arrays(
                None,
                None,
                timestamps_arr[micro_start:micro_end],
                feature_matrix[micro_start:micro_end],
                labels_arr[micro_start:micro_end],
                store,
                neighbor_k,
                feature_scaler=None,
                use_neighbors=use_neighbors,
                sender_ids=sender_ids[micro_start:micro_end] if sender_ids is not None else None,
                receiver_ids=receiver_ids[micro_start:micro_end] if receiver_ids is not None else None,
            )

        # Crucial causal boundary: no event at group_timestamp is visible to
        # any other query at the same timestamp.
        if use_neighbors and sender_ids is not None and receiver_ids is not None:
            _add_resolved_slice_to_store(
                sender_ids, receiver_ids, ts_list, feature_matrix, start, group_end, store
            )
        start = group_end


def populate_history(frame: pd.DataFrame, store: EventFeatureStore, batch_size: int, active_features=None, use_neighbors: bool = True) -> int:
    """Rebuild causal history without running the TGAT model."""
    active = list(active_features or ALL_FEATURES)
    timestamps_arr = frame["_timestamp"].to_numpy(dtype=np.int64)
    n_rows = len(timestamps_arr)
    if not use_neighbors or n_rows == 0:
        return 0
    ts_list = timestamps_arr.tolist()
    feature_matrix = np.ascontiguousarray(frame[active].to_numpy(dtype=np.float32))
    sender_ids = _resolve_account_ids(frame["From Account"].astype(str).tolist(), store)
    receiver_ids = _resolve_account_ids(frame["To Account"].astype(str).tolist(), store)

    start = 0
    batches = 0
    while start < n_rows:
        end = min(start + batch_size, n_rows)
        if end < n_rows:
            timestamp = ts_list[end - 1]
            while end < n_rows and ts_list[end] == timestamp:
                end += 1
        _add_resolved_slice_to_store(
            sender_ids, receiver_ids, ts_list, feature_matrix, start, end, store
        )
        batches += 1
        start = end
    return batches


def _summarize_grad_norms(norms: list[torch.Tensor], clip_limit: float) -> dict[str, float | int]:
    """Summarize asynchronous device-side step gradient norms on epoch completion."""
    if not norms:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "clip_count": 0,
            "clip_fraction": 0.0,
            "non_finite_count": 0,
        }
    stacked = torch.stack(norms).detach().cpu().numpy().astype(np.float64)
    finite_mask = np.isfinite(stacked)
    non_finite_count = int((~finite_mask).sum())
    valid = stacked[finite_mask]
    if len(valid) == 0:
        return {
            "count": int(len(stacked)),
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "clip_count": int(len(stacked)),
            "clip_fraction": 1.0,
            "non_finite_count": non_finite_count,
        }
    clipped = valid > clip_limit
    clip_count = int(clipped.sum()) + non_finite_count
    return {
        "count": int(len(stacked)),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "p50": float(np.quantile(valid, 0.50)),
        "p95": float(np.quantile(valid, 0.95)),
        "p99": float(np.quantile(valid, 0.99)),
        "max": float(valid.max()),
        "clip_count": clip_count,
        "clip_fraction": float(clip_count / len(stacked)),
        "non_finite_count": non_finite_count,
    }


def _score_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    def _q(arr: np.ndarray) -> dict[str, float | int]:
        if len(arr) == 0:
            return {"count": 0}
        return {
            "count": int(len(arr)),
            "min": float(arr.min()),
            "median": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "p999": float(np.quantile(arr, 0.999)),
            "max": float(arr.max()),
        }
    pos_max = float(pos.max()) if len(pos) else 0.0
    return {
        "pos": _q(pos),
        "neg": _q(neg),
        "neg_strictly_above_pos_max": int((neg > pos_max).sum()) if len(pos) else 0,
    }


def train_epoch(
    model,
    frame,
    store,
    optimizer,
    criterion,
    batch_size,
    neighbor_k,
    negative_ratio,
    device,
    active_features=None,
    use_neighbors=True,
    grad_clip_norm: float = 5.0,
    grad_stats_out: dict | None = None,
):
    model.train()
    total_loss_t = torch.zeros((), dtype=torch.float64, device=device)
    zero_scalar = torch.zeros((), dtype=torch.float32, device=device)
    pre_clip_grad_norms: list[torch.Tensor] = []
    te_grad_norms: list[torch.Tensor] = []
    freq_grad_norms: list[torch.Tensor] = []
    used = 0
    positives = 0
    negatives = 0
    total_batches = 0
    non_blocking = device.type == "cuda"

    timestamps_arr, ts_list, group_ends, feature_matrix, labels_arr, sender_ids, receiver_ids = _extract_split_arrays(
        frame, active_features, resolve_ids=use_neighbors, store=store
    )

    start = 0
    for group_end in group_ends:
        for micro_start in range(start, group_end, batch_size):
            micro_end = min(micro_start + batch_size, group_end)
            total_batches += 1
            labels = torch.from_numpy(labels_arr[micro_start:micro_end])
            positive_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
            negative_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
            keep_negative = min(len(negative_idx), max(1, len(positive_idx) * negative_ratio))
            if keep_negative < len(negative_idx):
                perm = torch.randperm(len(negative_idx))[:keep_negative]
                negative_idx = negative_idx[perm]
            keep = torch.cat([positive_idx, negative_idx])
            if len(keep) == 0:
                continue

            keep_np = keep.numpy()
            keep_list = keep_np.tolist()
            if use_neighbors and sender_ids is not None and receiver_ids is not None:
                batch_sender_ids = [sender_ids[micro_start + i] for i in keep_list]
                batch_receiver_ids = [receiver_ids[micro_start + i] for i in keep_list]
            else:
                batch_sender_ids = None
                batch_receiver_ids = None

            batch = build_batch_from_arrays(
                None,
                None,
                timestamps_arr[micro_start:micro_end][keep_np],
                feature_matrix[micro_start:micro_end][keep_np],
                labels_arr[micro_start:micro_end][keep_np],
                store,
                neighbor_k,
                feature_scaler=None,
                use_neighbors=use_neighbors,
                sender_ids=batch_sender_ids,
                receiver_ids=batch_receiver_ids,
            )

            transaction = batch.transaction_features.to(device, non_blocking=non_blocking)
            sender = batch.sender_features.to(device, non_blocking=non_blocking)
            sender_delta = batch.sender_delta_seconds.to(device, non_blocking=non_blocking)
            receiver = batch.receiver_features.to(device, non_blocking=non_blocking)
            receiver_delta = batch.receiver_delta_seconds.to(device, non_blocking=non_blocking)
            y = batch.labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(transaction, sender, sender_delta, receiver, receiver_delta)
            loss = criterion(logits, y)
            loss.backward()
            if grad_stats_out is not None:
                te_grads = [
                    p.grad.detach()
                    for p in model.time_encoder.parameters()
                    if p.grad is not None
                ]
                if te_grads:
                    te_norm = torch.linalg.vector_norm(
                        torch.cat([g.reshape(-1) for g in te_grads])
                    )
                else:
                    te_norm = zero_scalar
                te_grad_norms.append(te_norm)
                f_grad = model.time_encoder.frequency.grad
                freq_grad_norms.append(
                    torch.linalg.vector_norm(f_grad.detach()) if f_grad is not None else zero_scalar
                )
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if grad_stats_out is not None:
                pre_clip_grad_norms.append(total_norm.detach())
            optimizer.step()

            batch_count = len(keep)
            batch_pos = len(positive_idx)
            total_loss_t += loss.detach().double() * batch_count
            used += batch_count
            positives += batch_pos
            negatives += batch_count - batch_pos

            if total_batches % 100 == 0:
                print(f"  processed {total_batches} batches | queries={used:,} | pos={positives:,} | neg={negatives:,}", flush=True)

        if use_neighbors and sender_ids is not None and receiver_ids is not None:
            _add_resolved_slice_to_store(
                sender_ids, receiver_ids, ts_list, feature_matrix, start, group_end, store
            )
        start = group_end

    if grad_stats_out is not None:
        grad_stats_out.clear()
        grad_stats_out["pre_clip_grad_norm"] = _summarize_grad_norms(pre_clip_grad_norms, grad_clip_norm)
        grad_stats_out["time_encoder_grad_norm"] = _summarize_grad_norms(te_grad_norms, grad_clip_norm)
        grad_stats_out["time_frequency_grad_norm"] = _summarize_grad_norms(freq_grad_norms, grad_clip_norm)

    return float(total_loss_t.item()) / max(used, 1), positives, negatives, total_batches


def predict(model, frame, store, batch_size, neighbor_k, device, active_features=None, use_neighbors=True):
    model.eval()
    scores = []
    labels = []
    timestamps = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in make_batches(frame, store, batch_size, neighbor_k, active_features, use_neighbors):
            output = model(
                batch.transaction_features.to(device, non_blocking=non_blocking),
                batch.sender_features.to(device, non_blocking=non_blocking),
                batch.sender_delta_seconds.to(device, non_blocking=non_blocking),
                batch.receiver_features.to(device, non_blocking=non_blocking),
                batch.receiver_delta_seconds.to(device, non_blocking=non_blocking),
            )
            logits, _ = output
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch.labels.numpy())
            timestamps.append(batch.timestamps.numpy())
    if not scores:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    return (
        np.concatenate(labels).astype(np.int64, copy=False),
        np.concatenate(scores).astype(np.float64, copy=False),
        np.concatenate(timestamps).astype(np.int64, copy=False),
    )


def evaluate(labels, scores):
    return classification_metrics(labels, scores, threshold=0.5, k_values=(100, 500, 1000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--tail-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ablation", choices=["none","no_temporal_features","no_neighbors","no_time_encoding"], default="none")
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, choices=[1,2], default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-encoding-mode",
        choices=list(VALID_TIME_ENCODING_MODES),
        default="raw_seconds",
    )
    parser.add_argument("--time-max-period", type=float, default=86_400.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    if args.ablation == "no_temporal_features":
        active_features = list(CURRENT_FEATURES)
    else:
        active_features = list(ALL_FEATURES)
    use_neighbors = args.ablation != "no_neighbors"
    use_time_encoding = args.ablation != "no_time_encoding"
    max_history = args.max_history if args.max_history is not None else args.neighbor_k

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = 5000 if args.smoke else args.max_train_rows
    tail_train_rows = None if args.smoke else args.tail_train_rows
    val_rows = 2000 if args.smoke else args.max_validation_rows
    test_rows = 2000 if args.smoke else args.max_test_rows

    train = read_frame(args.data_dir / "train.csv", train_rows, tail_rows=tail_train_rows)
    validation = read_frame(args.data_dir / "validation.csv", val_rows)
    test = None if args.skip_test else read_frame(args.data_dir / "test.csv", test_rows)
    if len(train) == 0 or len(validation) == 0 or (test is not None and len(test) == 0):
        raise RuntimeError("TGAT input split is empty")
    if list(train[ALL_FEATURES].columns) != ALL_FEATURES:
        raise RuntimeError("TGAT feature contract mismatch")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("Invalid target labels")

    # Fit statistics on training data only. Transform each split once and keep
    # both query features and stored historical event features in that same
    # scaled space. No future information enters the scaler.
    print("Fitting train-only feature scaler...", flush=True)
    scaler = TemporalFeatureScaler.fit(train)
    train = scaler.transform_frame(train)
    gc.collect()
    validation = scaler.transform_frame(validation)
    if test is not None:
        test = scaler.transform_frame(test)
    keep_cols = {"From Account", "To Account", "_timestamp", TARGET, *active_features}
    for frame_ref in (train, validation, test):
        if frame_ref is not None:
            drop_cols = [c for c in frame_ref.columns if c not in keep_cols]
            if drop_cols:
                frame_ref.drop(columns=drop_cols, inplace=True)
            for col in active_features:
                if frame_ref[col].dtype != np.float32:
                    frame_ref[col] = frame_ref[col].astype(np.float32, copy=False)
    gc.collect()
    print("Feature scaling: train-only mean/std, clipped to [-10, 10]", flush=True)

    model = TGATTransactionModel(
        transaction_dim=len(active_features),
        event_dim=len(active_features),
        hidden_dim=args.hidden_dim,
        time_dim=16,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        use_time_encoding=use_time_encoding,
        time_encoding_mode=args.time_encoding_mode,
        max_period=args.time_max_period,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "scaler_params.json").write_text(
        json.dumps(
            {
                "features": list(scaler.features),
                "mean": scaler.mean.tolist(),
                "scale": scaler.scale.tolist(),
                "clip": float(scaler.clip),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model_parameters = sum(p.numel() for p in model.parameters())
    timestamp_groups = {
        "train": int(train["_timestamp"].nunique()),
        "validation": int(validation["_timestamp"].nunique()),
        "test": int(test["_timestamp"].nunique()) if test is not None else 0,
    }
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "seed": args.seed,
        "feature_count": len(active_features),
        "active_features": active_features,
        "ablation": args.ablation,
        "use_neighbors": use_neighbors,
        "use_time_encoding": use_time_encoding,
        "time_encoding_mode": args.time_encoding_mode,
        "time_max_period": args.time_max_period,
        "grad_clip_norm": args.grad_clip_norm,
        "max_history": max_history,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_parameters": model_parameters,
        "current_feature_count": len(CURRENT_FEATURES),
        "temporal_feature_count": len(active_features) - len(CURRENT_FEATURES),
        "timestamp_unit": "seconds",
        "timestamp_resolution_seconds": 1,
        "timestamp_groups": timestamp_groups,
        "feature_scaling": "train_only_mean_std_clip_10",
    })
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    best_val = -np.inf
    best_epoch = 0
    best_state = None
    log = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        store = EventFeatureStore(max_history=max_history)
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs} - training...", flush=True)
        grad_stats: dict = {}
        loss, positives, negatives, batches = train_epoch(
            model, train, store, optimizer, criterion,
            args.batch_size, args.neighbor_k, args.negative_ratio, device,
            active_features, use_neighbors,
            grad_clip_norm=args.grad_clip_norm,
            grad_stats_out=grad_stats,
        )
        print(f"Epoch {epoch}/{args.epochs} - validating...", flush=True)
        val_labels, val_scores, _ = predict(
            model, validation, store, args.batch_size, args.neighbor_k, device, active_features, use_neighbors
        )
        val_pr = average_precision_score(val_labels, val_scores)
        with torch.no_grad():
            freq_np = model.time_encoder.frequency.detach().cpu().numpy().astype(np.float64)
            phase_np = model.time_encoder.phase.detach().cpu().numpy().astype(np.float64)
        g_stats = grad_stats["pre_clip_grad_norm"]
        te_stats = grad_stats["time_encoder_grad_norm"]
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_positive_queries": positives,
            "train_negative_queries": negatives,
            "train_batches": batches,
            "validation_pr_auc": float(val_pr),
            "seconds": time.time() - epoch_start,
            "pre_clip_grad_norm": g_stats,
            "time_encoder_grad_norm": te_stats,
            "time_frequency_grad_norm": grad_stats["time_frequency_grad_norm"],
            "time_encoder_state": {
                "frequency_min": float(freq_np.min()),
                "frequency_max": float(freq_np.max()),
                "frequency_l2": float(np.linalg.norm(freq_np)),
                "phase_l2": float(np.linalg.norm(phase_np)),
                "frequency_values": freq_np.tolist(),
                "phase_values": phase_np.tolist(),
            },
        }
        log.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={loss:.6f}, "
            f"val_pr_auc={val_pr:.6f}, grad_mean={g_stats['mean']:.3f}, "
            f"grad_p95={g_stats['p95']:.3f}, grad_max={g_stats['max']:.3f}, "
            f"clip_frac={g_stats['clip_fraction']*100:.2f}%, "
            f"te_grad_mean={te_stats['mean']:.3f}, time={row['seconds']:.1f}s",
            flush=True,
        )
        if val_pr > best_val:
            best_val = val_pr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No TGAT checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    (args.results_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nRebuilding train history for final validation/test evaluation...", flush=True)
    evaluation_store = EventFeatureStore(max_history=max_history)
    history_start = time.time()
    history_batches = populate_history(
        train, evaluation_store, args.batch_size, active_features, use_neighbors=use_neighbors
    )
    print(
        f"Train history rebuilt: {len(train):,} rows, "
        f"{history_batches} batches, time={time.time() - history_start:.1f}s",
        flush=True,
    )
    print("Evaluating validation...", flush=True)
    val_labels, val_scores, val_ts = predict(
        model, validation, evaluation_store, args.batch_size, args.neighbor_k, device, active_features, use_neighbors
    )
    if test is not None:
        print("Evaluating test...", flush=True)
        test_labels, test_scores, test_ts = predict(
            model, test, evaluation_store, args.batch_size, args.neighbor_k, device, active_features, use_neighbors
        )
        test_eval = evaluate(test_labels, test_scores)
        test_summary = _score_summary(test_labels, test_scores)
    else:
        print("Skipping test split evaluation (--skip-test enabled).", flush=True)
        test_labels = test_scores = test_ts = None
        test_eval = None
        test_summary = None

    total_opt_steps = sum(int(r["pre_clip_grad_norm"]["count"]) for r in log)
    total_clip_steps = sum(int(r["pre_clip_grad_norm"]["clip_count"]) for r in log)
    metrics = {
        "validation": evaluate(val_labels, val_scores),
        "test": test_eval,
        "validation_score_summary": _score_summary(val_labels, val_scores),
        "test_score_summary": test_summary,
        "test_evaluated": test is not None,
        "selected_epoch": best_epoch,
        "best_validation_pr_auc": float(best_val),
        "training_stability": {
            "total_optimizer_steps": total_opt_steps,
            "total_clipped_steps": total_clip_steps,
            "overall_clip_fraction": float(total_clip_steps / total_opt_steps) if total_opt_steps else 0.0,
            "epoch_losses": [float(r["loss"]) for r in log],
            "epoch_grad_norm_means": [float(r["pre_clip_grad_norm"]["mean"]) for r in log],
            "epoch_grad_norm_maxes": [float(r["pre_clip_grad_norm"]["max"]) for r in log],
            "epoch_clip_fractions": [float(r["pre_clip_grad_norm"]["clip_fraction"]) for r in log],
            "epoch_te_grad_norm_means": [float(r["time_encoder_grad_norm"]["mean"]) for r in log],
        },
        "training_seconds": time.time() - start_time,
        "seed": args.seed,
        "feature_count": len(active_features),
        "neighbor_k": args.neighbor_k,
        "max_history": max_history,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip_norm": args.grad_clip_norm,
        "time_encoding_mode": args.time_encoding_mode,
        "time_max_period": args.time_max_period,
        "model_parameters": model_parameters,
        "ablation": args.ablation,
        "negative_sampling_ratio": args.negative_ratio,
        "device": str(device),
        "timestamp_unit": "seconds",
        "timestamp_resolution_seconds": 1,
        "timestamp_groups": timestamp_groups,
        "feature_scaling": "train_only_mean_std_clip_10",
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"timestamp": val_ts, "label": val_labels, "score": val_scores}).to_csv(
        args.results_dir / "validation_predictions.csv", index=False
    )
    if test_labels is not None:
        pd.DataFrame({"timestamp": test_ts, "label": test_labels, "score": test_scores}).to_csv(
            args.results_dir / "test_predictions.csv", index=False
        )
    print("TGAT TRAINING PIPELINE: COMPLETED")


if __name__ == "__main__":
    main()
