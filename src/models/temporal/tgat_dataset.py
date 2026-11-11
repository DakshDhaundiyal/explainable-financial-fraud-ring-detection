"""Build compact chronological TGAT batches from temporal GNN datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .temporal_neighbor_store import TemporalNeighborStore

CURRENT_FEATURES = [
    "Amount Received", "Amount Paid", "Amount Difference", "Amount Ratio",
    "Same Bank Transaction", "Cross Bank Transaction", "Transaction Time Category",
    "Is Weekend", "Log Amount Received", "Log Amount Paid", "Same Currency",
]
TEMPORAL_FEATURES = [
    "sender_in_count", "sender_out_count", "sender_total_count", "sender_in_amount",
    "sender_out_amount", "sender_avg_in_amount", "sender_avg_out_amount", "sender_time_since_last",
    "receiver_in_count", "receiver_out_count", "receiver_total_count", "receiver_in_amount",
    "receiver_out_amount", "receiver_avg_in_amount", "receiver_time_since_last",
]
# `receiver_avg_out_amount` is present in temporal_gnn/*.csv (27 features) and
# used by Temporal XGBoost, but excluded from the 26-feature TGAT contract.
OMITTED_CSV_TEMPORAL_FEATURES = ["receiver_avg_out_amount"]
ALL_FEATURES = CURRENT_FEATURES + TEMPORAL_FEATURES
FEATURE_DIM = len(ALL_FEATURES)
TARGET = "Is Laundering"


def to_unix_seconds(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Convert parsed datetime values explicitly to 1-second Unix epoch integers.

    Under pandas 3.x parsed timestamps default to ``datetime64[us]`` rather than
    ``datetime64[ns]``, so dividing raw ``int64`` values by ``10**9`` collapses
    timestamps into 1,000-second buckets. Converting via ``as_unit('s')`` preserves
    exact 1-second resolution independent of the underlying datetime unit.
    """
    series = pd.Series(values) if isinstance(values, pd.DatetimeIndex) else values
    if series.isna().any():
        raise ValueError("Cannot convert NaT timestamps to Unix seconds")
    return series.dt.as_unit("s").astype("int64")


@dataclass
class TemporalBatch:
    transaction_features: torch.Tensor
    sender_features: torch.Tensor
    sender_delta_seconds: torch.Tensor
    receiver_features: torch.Tensor
    receiver_delta_seconds: torch.Tensor
    labels: torch.Tensor
    timestamps: torch.Tensor


class EventFeatureStore:
    """Stores compact event feature vectors for temporal neighbor lookup."""

    def __init__(self, max_history: int = 64) -> None:
        self.store = TemporalNeighborStore(max_history_per_node=max_history)
        self.account_to_id: dict[str, int] = {}

    def get_id(self, account: str) -> int:
        if account not in self.account_to_id:
            self.account_to_id[account] = len(self.account_to_id)
        return self.account_to_id[account]

    def add_row(self, row, timestamp: int, feature_indices: tuple[int, ...]) -> None:
        """Insert one itertuples row without constructing pandas objects."""
        source = self.get_id(str(row[feature_indices[0]]))
        destination = self.get_id(str(row[feature_indices[1]]))
        self.store.add_event(
            timestamp,
            source,
            destination,
            tuple(row[index] for index in feature_indices[2:]),
        )

    def history(self, node_id: int, timestamp: int, limit: int) -> list[tuple[int, np.ndarray]]:
        events = self.store.recent_events(node_id, timestamp, limit)
        return [(event.timestamp, np.asarray(event.features, dtype=np.float32)) for event in events]


def _pad_history(events: Iterable[tuple[int, np.ndarray]], k: int, feature_dim: int, query_timestamp: int) -> tuple[np.ndarray, np.ndarray]:
    feature_array = np.zeros((k, feature_dim), dtype=np.float32)
    delta_array = np.zeros(k, dtype=np.float32)
    for index, (timestamp, features) in enumerate(events):
        if index >= k:
            break
        feature_array[index] = features
        delta = int(query_timestamp) - int(timestamp)
        if delta <= 0:
            raise ValueError("Temporal sampler returned a non-historical event")
        delta_array[index] = float(delta)
    return feature_array, delta_array


def build_batch_from_arrays(
    senders: np.ndarray | list[str] | None,
    receivers: np.ndarray | list[str] | None,
    timestamps_arr: np.ndarray,
    feature_matrix: np.ndarray,
    labels_arr: np.ndarray,
    store: EventFeatureStore,
    neighbor_k: int = 10,
    feature_scaler=None,
    use_neighbors: bool = True,
    sender_ids: np.ndarray | list[int] | None = None,
    receiver_ids: np.ndarray | list[int] | None = None,
) -> TemporalBatch:
    """Build a causal TemporalBatch from pre-extracted arrays."""
    batch_len = len(timestamps_arr)
    feature_dim = feature_matrix.shape[1]

    if feature_scaler is not None:
        tx_out = feature_scaler.transform_array(feature_matrix).astype(np.float32, copy=False)
    else:
        tx_out = np.asarray(feature_matrix, dtype=np.float32)

    sender_buf = np.zeros((batch_len, neighbor_k + 1, feature_dim), dtype=np.float32)
    receiver_buf = np.zeros((batch_len, neighbor_k + 1, feature_dim), dtype=np.float32)
    sender_delta_buf = np.zeros((batch_len, neighbor_k), dtype=np.float32)
    receiver_delta_buf = np.zeros((batch_len, neighbor_k), dtype=np.float32)

    if use_neighbors:
        get_id = store.get_id
        recent_events = store.store.recent_events
        ts_list = timestamps_arr.tolist() if isinstance(timestamps_arr, np.ndarray) else timestamps_arr
        if sender_ids is not None and receiver_ids is not None:
            s_id_iter = sender_ids.tolist() if isinstance(sender_ids, np.ndarray) else sender_ids
            r_id_iter = receiver_ids.tolist() if isinstance(receiver_ids, np.ndarray) else receiver_ids
        else:
            s_id_iter = [get_id(str(senders[row_idx])) for row_idx in range(batch_len)]
            r_id_iter = [get_id(str(receivers[row_idx])) for row_idx in range(batch_len)]

        for row_idx in range(batch_len):
            timestamp = int(ts_list[row_idx])
            s_events = recent_events(int(s_id_iter[row_idx]), timestamp, neighbor_k)
            for j, ev in enumerate(s_events):
                if j >= neighbor_k:
                    break
                delta = timestamp - ev.timestamp
                if delta <= 0:
                    raise ValueError("Temporal sampler returned a non-historical event")
                sender_buf[row_idx, j + 1] = ev.features
                sender_delta_buf[row_idx, j] = float(delta)

            r_events = recent_events(int(r_id_iter[row_idx]), timestamp, neighbor_k)
            for j, ev in enumerate(r_events):
                if j >= neighbor_k:
                    break
                delta = timestamp - ev.timestamp
                if delta <= 0:
                    raise ValueError("Temporal sampler returned a non-historical event")
                receiver_buf[row_idx, j + 1] = ev.features
                receiver_delta_buf[row_idx, j] = float(delta)

    return TemporalBatch(
        transaction_features=torch.from_numpy(np.ascontiguousarray(tx_out.copy())),
        sender_features=torch.from_numpy(sender_buf),
        sender_delta_seconds=torch.from_numpy(sender_delta_buf),
        receiver_features=torch.from_numpy(receiver_buf),
        receiver_delta_seconds=torch.from_numpy(receiver_delta_buf),
        labels=torch.from_numpy(np.ascontiguousarray(np.asarray(labels_arr, dtype=np.float32).copy())),
        timestamps=torch.from_numpy(np.ascontiguousarray(np.asarray(timestamps_arr, dtype=np.int64).copy())),
    )


def build_batch(frame: pd.DataFrame, store: EventFeatureStore, neighbor_k: int = 10, feature_scaler=None, active_features: list[str] | None = None, use_neighbors: bool = True) -> TemporalBatch:
    """Build a causal batch using tuple-based row access."""
    columns = list(frame.columns)
    idx = {name: pos for pos, name in enumerate(columns)}
    active_features = list(active_features or ALL_FEATURES)
    feature_indices = tuple(idx[name] for name in active_features)
    sender_index = idx["From Account"]
    receiver_index = idx["To Account"]
    timestamp_index = idx["_timestamp"]
    target_index = idx[TARGET]
    transaction_features, sender_features, sender_delta = [], [], []
    receiver_features, receiver_delta, labels, timestamps = [], [], [], []

    for row in frame.itertuples(index=False, name=None):
        timestamp = int(row[timestamp_index])
        sender = store.get_id(str(row[sender_index]))
        receiver = store.get_id(str(row[receiver_index]))
        if use_neighbors:
            sender_events, sender_times = _pad_history(
                store.history(sender, timestamp, neighbor_k), neighbor_k, len(active_features), timestamp
            )
            receiver_events, receiver_times = _pad_history(
                store.history(receiver, timestamp, neighbor_k), neighbor_k, len(active_features), timestamp
            )
        else:
            sender_events = np.zeros((neighbor_k, len(active_features)), dtype=np.float32)
            receiver_events = np.zeros((neighbor_k, len(active_features)), dtype=np.float32)
            sender_times = np.zeros(neighbor_k, dtype=np.float32)
            receiver_times = np.zeros(neighbor_k, dtype=np.float32)
        transaction = np.asarray([row[index] for index in feature_indices], dtype=np.float64)
        if feature_scaler is not None:
            transaction = feature_scaler.transform_array(transaction)
        else:
            transaction = transaction.astype(np.float32)
        transaction_features.append(transaction)
        sender_features.append(np.vstack([np.zeros(len(active_features), dtype=np.float32), sender_events]))
        receiver_features.append(np.vstack([np.zeros(len(active_features), dtype=np.float32), receiver_events]))
        sender_delta.append(sender_times)
        receiver_delta.append(receiver_times)
        labels.append(int(row[target_index]))
        timestamps.append(timestamp)

    return TemporalBatch(
        transaction_features=torch.from_numpy(np.stack(transaction_features)),
        sender_features=torch.from_numpy(np.stack(sender_features)),
        sender_delta_seconds=torch.from_numpy(np.stack(sender_delta)),
        receiver_features=torch.from_numpy(np.stack(receiver_features)),
        receiver_delta_seconds=torch.from_numpy(np.stack(receiver_delta)),
        labels=torch.tensor(labels, dtype=torch.float32),
        timestamps=torch.tensor(timestamps, dtype=torch.long),
    )


class TGATSmokeDataset(Dataset):
    """Small deterministic dataset wrapper used by tests."""
    def __init__(self, frame: pd.DataFrame, neighbor_k: int = 10) -> None:
        self.frame = frame.reset_index(drop=True)
        self.neighbor_k = neighbor_k
    def __len__(self) -> int:
        return len(self.frame)
    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        row = self.frame.iloc[index]
        return {"transaction": row[ALL_FEATURES].to_numpy(dtype=np.float32), "timestamp": int(row["_timestamp"]), "label": int(row[TARGET])}