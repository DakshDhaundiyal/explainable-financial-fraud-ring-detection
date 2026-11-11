"""Chronological interaction store for leakage-safe temporal neighborhoods."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """One historical account interaction."""
    timestamp: int
    source: int
    destination: int
    features: tuple[float, ...] | np.ndarray


class TemporalNeighborStore:
    """In-memory append-only event history with strict timestamp queries.

    Events are appended in chronological order. Each per-node deque is kept in
    ascending timestamp order. Queries use binary search to find the prefix
    strictly earlier than the query timestamp, then scan only that prefix
    backwards to collect the most recent events.
    """

    def __init__(self, max_history_per_node: int = 64) -> None:
        if max_history_per_node <= 0:
            raise ValueError("max_history_per_node must be positive")
        self.max_history_per_node = max_history_per_node
        bounded_deque = partial(deque, maxlen=max_history_per_node)
        self._incoming: dict[int, deque[TemporalEvent]] = defaultdict(bounded_deque)
        self._outgoing: dict[int, deque[TemporalEvent]] = defaultdict(bounded_deque)
        self._incoming_timestamps: dict[int, deque[int]] = defaultdict(bounded_deque)
        self._outgoing_timestamps: dict[int, deque[int]] = defaultdict(bounded_deque)

    def add_event(
        self,
        timestamp: int,
        source: int,
        destination: int,
        features: Iterable[float],
    ) -> None:
        if isinstance(features, np.ndarray) and features.dtype == np.float32 and features.ndim == 1:
            stored_features: tuple[float, ...] | np.ndarray = features
        else:
            stored_features = tuple(float(v) for v in features)

        ts_int = int(timestamp)
        src_int = int(source)
        dst_int = int(destination)
        event = TemporalEvent(
            timestamp=ts_int,
            source=src_int,
            destination=dst_int,
            features=stored_features,
        )
        self._outgoing[src_int].append(event)
        self._outgoing_timestamps[src_int].append(ts_int)
        self._incoming[dst_int].append(event)
        self._incoming_timestamps[dst_int].append(ts_int)

    @staticmethod
    def _recent_before(
        history: deque[TemporalEvent],
        timestamps: deque[int],
        query_timestamp: int,
        limit: int,
    ) -> list[TemporalEvent]:
        """Return up to limit most-recent events with timestamp < query.

        Binary search identifies the last eligible position. This avoids
        scanning recent events that are at the query timestamp, which is common
        in this dataset because many transactions share exact timestamps.
        """
        if not history:
            return []
        eligible_end = bisect_left(timestamps, int(query_timestamp))
        if eligible_end <= 0:
            return []
        start = max(0, eligible_end - int(limit))
        return [history[idx] for idx in range(eligible_end - 1, start - 1, -1)]

    def get_history(
        self,
        node_id: int,
        query_timestamp: int,
        limit: int,
    ) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
        """Return recent outgoing and incoming events strictly before T."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        node_id = int(node_id)
        ts_query = int(query_timestamp)
        limit_int = int(limit)

        out_hist = self._outgoing.get(node_id)
        outgoing = (
            self._recent_before(
                out_hist,
                self._outgoing_timestamps[node_id],
                ts_query,
                limit_int,
            )
            if out_hist
            else []
        )

        in_hist = self._incoming.get(node_id)
        incoming = (
            self._recent_before(
                in_hist,
                self._incoming_timestamps[node_id],
                ts_query,
                limit_int,
            )
            if in_hist
            else []
        )
        return outgoing, incoming

    def recent_events(
        self,
        node_id: int,
        query_timestamp: int,
        limit: int,
    ) -> list[TemporalEvent]:
        """Return a merged recent temporal neighborhood strictly before T."""
        outgoing, incoming = self.get_history(node_id, query_timestamp, limit)
        if not incoming:
            return outgoing[:limit]
        if not outgoing:
            return incoming[:limit]
        events = outgoing + incoming
        events.sort(key=lambda item: item.timestamp, reverse=True)
        return events[:limit]

    def __len__(self) -> int:
        nodes = set(self._incoming) | set(self._outgoing)
        return len(nodes)
