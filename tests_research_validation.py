"""Regression and numerical verification tests for Day 1 research validation fixes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.temporal.build_temporal_features import build_temporal_features
from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.temporal_dataset import (
    CURRENT_TRANSACTION_FEATURES,
    OUTPUT_COLUMNS,
    process_split,
)
from src.models.temporal.temporal_state import TemporalStateStore
from src.models.temporal.tgat_core import TGATTransactionModel, TimeEncoder
from src.models.temporal.tgat_dataset import (
    ALL_FEATURES,
    CURRENT_FEATURES,
    FEATURE_DIM,
    OMITTED_CSV_TEMPORAL_FEATURES,
    TARGET,
    EventFeatureStore,
    build_batch,
    to_unix_seconds,
)
from src.models.temporal.train_tgat import make_batches, populate_history, predict, read_frame, seed_everything, train_epoch


def test_pandas_microsecond_quantization_repro_and_to_unix_seconds_fix() -> None:
    stamps = [
        "2022/09/01 00:15:00",
        "2022/09/01 00:15:30",
        "2022/09/01 00:16:00",
        "2022/09/01 00:16:39",
    ]
    parsed = pd.Series(pd.to_datetime(stamps, format="mixed", errors="raise"))
    if parsed.dtype == "datetime64[us]":
        # Reproduce the pandas 3.x microsecond-to-1000s collapse bug explicitly.
        buggy = parsed.astype("int64") // 1_000_000_000
        assert buggy.nunique() == 1
        assert int(buggy.iloc[0]) == 1661991

    # Verify the explicit 1-second converter works identically across units and DatetimeIndex.
    expected = [1661991300, 1661991330, 1661991360, 1661991399]
    assert to_unix_seconds(parsed).tolist() == expected
    assert to_unix_seconds(pd.to_datetime(stamps, format="mixed")).tolist() == expected
    for unit in ("s", "ms", "us", "ns"):
        series_unit = pd.Series(pd.to_datetime(stamps, format="mixed")).dt.as_unit(unit)
        converted = to_unix_seconds(series_unit)
        assert converted.tolist() == expected, f"Failed for unit={unit}"

    # NaT guard must raise explicitly.
    bad = pd.Series(pd.to_datetime(["2022-09-01 00:00:00", "not-a-date"], errors="coerce"))
    try:
        to_unix_seconds(bad)
        raise AssertionError("Expected ValueError on NaT")
    except ValueError:
        pass


def _make_synthetic_split_csv(path: Path) -> None:
    rows = [
        # T0 = 00:00:00
        ("2022/09/01 00:00:00", "A", "B", 100.0, 100.0, 0),
        # T1 = 00:01:00 (two concurrent edges; must both see T0 only)
        ("2022/09/01 00:01:00", "A", "C", 250.0, 250.0, 1),
        ("2022/09/01 00:01:00", "B", "A", 50.0, 50.0, 0),
        # T2 = 00:02:30 (90s after T1, 150s after T0)
        ("2022/09/01 00:02:30", "A", "D", 75.0, 75.0, 0),
        ("2022/09/01 00:02:30", "C", "D", 40.0, 40.0, 0),
    ]
    records = []
    for i, (ts, src, dst, amt_r, amt_p, y) in enumerate(rows):
        rec = {
            "Timestamp": ts,
            "From Account": src,
            "To Account": dst,
            "Amount Received": amt_r,
            "Amount Paid": amt_p,
            "Amount Difference": 0.0,
            "Amount Ratio": 1.0,
            "Same Bank Transaction": 1 if i % 2 == 0 else 0,
            "Cross Bank Transaction": 0 if i % 2 == 0 else 1,
            "Transaction Time Category": 0,
            "Is Weekend": 0,
            "Log Amount Received": float(np.log1p(amt_r)),
            "Log Amount Paid": float(np.log1p(amt_p)),
            "Same Currency": 1,
            "Is Laundering": y,
        }
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False)


def test_temporal_dataset_one_second_time_since_last_and_chunk_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_csv = tmp_dir / "input.csv"
        out_small_chunk = tmp_dir / "out_chunk2.csv"
        out_large_chunk = tmp_dir / "out_chunk100.csv"
        _make_synthetic_split_csv(in_csv)

        rows_2, groups_2 = process_split(in_csv, out_small_chunk, TemporalStateStore(), chunk_size=2)
        rows_100, groups_100 = process_split(in_csv, out_large_chunk, TemporalStateStore(), chunk_size=100)

        assert rows_2 == 5 and rows_100 == 5
        assert groups_2 == 3 and groups_100 == 3

        df_2 = pd.read_csv(out_small_chunk)
        df_100 = pd.read_csv(out_large_chunk)
        assert list(df_2.columns) == OUTPUT_COLUMNS
        pd.testing.assert_frame_equal(df_2, df_100)

        # Row 0 (T0=00:00:00, A->B): no prior state
        assert df_2.loc[0, "sender_total_count"] == 0.0
        assert df_2.loc[0, "sender_time_since_last"] == 0.0
        assert df_2.loc[0, "receiver_time_since_last"] == 0.0

        # Rows 1 and 2 (both T1=00:01:00): both strictly see only T0 pre-state (60s gap)
        # Concurrent B->A at row 2 must NOT mutate A before row 1, nor A->C mutate A before row 2.
        assert df_2.loc[1, "sender_out_count"] == 1.0
        assert df_2.loc[1, "sender_in_count"] == 0.0
        assert df_2.loc[1, "sender_time_since_last"] == 60.0
        assert df_2.loc[1, "receiver_total_count"] == 0.0

        assert df_2.loc[2, "sender_in_count"] == 1.0
        assert df_2.loc[2, "sender_out_count"] == 0.0
        assert df_2.loc[2, "sender_time_since_last"] == 60.0
        assert df_2.loc[2, "receiver_out_count"] == 1.0
        assert df_2.loc[2, "receiver_in_count"] == 0.0
        assert df_2.loc[2, "receiver_time_since_last"] == 60.0

        # Row 3 (T2=00:02:30, A->D): 90s after T1 (A had 1 out at T0, 1 out + 1 in at T1)
        assert df_2.loc[3, "sender_out_count"] == 2.0
        assert df_2.loc[3, "sender_in_count"] == 1.0
        assert df_2.loc[3, "sender_time_since_last"] == 90.0
        assert df_2.loc[4, "sender_in_count"] == 1.0
        assert df_2.loc[4, "sender_time_since_last"] == 90.0

        # Cross-chunk out-of-order check must fail rather than violate causality:
        # Chunk 1 ends at T2 (row 3), while Chunk 2 starts at T1 (row 1 < T2).
        bad_csv = tmp_dir / "bad_order.csv"
        raw = pd.read_csv(in_csv)
        raw.iloc[[0, 3, 1, 2, 4]].to_csv(bad_csv, index=False)
        try:
            process_split(bad_csv, tmp_dir / "bad_out.csv", TemporalStateStore(), chunk_size=2)
            raise AssertionError("Expected ValueError for non-chronological chunk boundary")
        except ValueError:
            pass


def test_build_temporal_features_matches_one_second_unix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        in_csv = Path(tmp) / "input.csv"
        _make_synthetic_split_csv(in_csv)
        built = build_temporal_features(str(in_csv))
        assert built["sender_time_since_last"].tolist() == [0.0, 60.0, 60.0, 90.0, 90.0]
        assert built["receiver_time_since_last"].tolist() == [0.0, 0.0, 60.0, 0.0, 0.0]


def test_time_encoder_distinguishes_one_second_and_minute_deltas() -> None:
    encoder = TimeEncoder(dimension=16)
    with torch.no_grad():
        collapsed = encoder(torch.tensor([[0.0], [0.0], [0.0]]))
        assert torch.allclose(collapsed[0], collapsed[1])
        assert torch.allclose(collapsed[1], collapsed[2])

        resolved = encoder(torch.tensor([[60.0], [300.0], [600.0]]))
        assert torch.isfinite(resolved).all()
        assert float((resolved[0] - resolved[1]).abs().max()) > 1e-2
        assert float((resolved[1] - resolved[2]).abs().max()) > 1e-2


def test_event_store_and_make_batches_causality_and_reference_parity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "temporal.csv"
        rows = []
        base_ts = pd.Timestamp("2022-09-01 00:00:00")
        specs = [
            (0, "A", "B", 10.0, 0),
            (0, "B", "C", 20.0, 1),
            (0, "A", "C", 30.0, 0),
            (60, "A", "B", 40.0, 0),
            (60, "B", "A", 50.0, 1),
            (120, "A", "D", 60.0, 0),
            (180, "A", "B", 70.0, 0),
            (240, "A", "B", 80.0, 1),
            (300, "A", "B", 90.0, 0),
            (360, "A", "B", 100.0, 0),
            (420, "A", "B", 110.0, 0),
            (480, "A", "B", 120.0, 1),
            (540, "A", "B", 130.0, 0),
            (600, "A", "C", 140.0, 0),
        ]
        for offset, src, dst, amt, y in specs:
            rec = {
                "From Account": src,
                "To Account": dst,
                "Timestamp": (base_ts + pd.Timedelta(seconds=offset)).strftime("%Y/%m/%d %H:%M:%S"),
                "Amount Received": amt,
                "Amount Paid": amt,
                "Amount Difference": 0.0,
                "Amount Ratio": 1.0,
                "Same Bank Transaction": 1,
                "Cross Bank Transaction": 0,
                "Transaction Time Category": 0,
                "Is Weekend": 0,
                "Log Amount Received": float(np.log1p(amt)),
                "Log Amount Paid": float(np.log1p(amt)),
                "Same Currency": 1,
                "sender_in_count": float(offset // 60),
                "sender_out_count": 1.0,
                "sender_total_count": 2.0,
                "sender_in_amount": amt,
                "sender_out_amount": amt,
                "sender_avg_in_amount": amt,
                "sender_avg_out_amount": amt,
                "sender_time_since_last": 60.0 if offset > 0 else 0.0,
                "receiver_in_count": 1.0,
                "receiver_out_count": 0.0,
                "receiver_total_count": 1.0,
                "receiver_in_amount": amt,
                "receiver_out_amount": 0.0,
                "receiver_avg_in_amount": amt,
                "receiver_avg_out_amount": 0.0,
                "receiver_time_since_last": 60.0 if offset > 0 else 0.0,
                "Is Laundering": y,
            }
            rows.append(rec)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        frame = read_frame(out_csv)
        scaler = TemporalFeatureScaler.fit(frame)
        scaled = scaler.transform_frame(frame)

        fast_store = EventFeatureStore(max_history=5)
        fast_batches = list(make_batches(scaled, fast_store, batch_size=2, neighbor_k=5))

        # Reference group-by-timestamp build using build_batch + add_row
        ref_store = EventFeatureStore(max_history=5)
        ref_batches = []
        cols = list(scaled.columns)
        idx_map = {name: pos for pos, name in enumerate(cols)}
        store_indices = (
            idx_map["From Account"],
            idx_map["To Account"],
            *(idx_map[col] for col in ALL_FEATURES),
        )
        for _, grp in scaled.groupby("_timestamp", sort=False):
            for i in range(0, len(grp), 2):
                chunk = grp.iloc[i : i + 2]
                ref_batches.append(build_batch(chunk, ref_store, neighbor_k=5, use_neighbors=True))
            for row in grp.itertuples(index=False, name=None):
                ref_store.add_row(row, int(row[idx_map["_timestamp"]]), feature_indices=store_indices)

        assert len(fast_batches) == len(ref_batches)
        for fb, rb in zip(fast_batches, ref_batches):
            assert torch.equal(fb.transaction_features, rb.transaction_features)
            assert torch.equal(fb.sender_features, rb.sender_features)
            assert torch.equal(fb.sender_delta_seconds, rb.sender_delta_seconds)
            assert torch.equal(fb.receiver_features, rb.receiver_features)
            assert torch.equal(fb.receiver_delta_seconds, rb.receiver_delta_seconds)
            assert torch.equal(fb.labels, rb.labels)
            assert torch.equal(fb.timestamps, rb.timestamps)
            # Check strict positivity on every non-zero neighbor delta
            s_nonzero = fb.sender_delta_seconds > 0
            assert (fb.sender_delta_seconds[~s_nonzero] == 0).all()
            r_nonzero = fb.receiver_delta_seconds > 0
            assert (fb.receiver_delta_seconds[~r_nonzero] == 0).all()

        # First timestamp has 3 rows split into batches of size 2 + 1: both must have zero neighbor history
        assert float(fast_batches[0].sender_delta_seconds.sum()) == 0.0
        assert float(fast_batches[1].sender_delta_seconds.sum()) == 0.0
        assert float(fast_batches[2].sender_delta_seconds.sum()) > 0.0


def test_feature_contract_26_synchronized() -> None:
    assert len(ALL_FEATURES) == 26
    assert FEATURE_DIM == 26
    assert OMITTED_CSV_TEMPORAL_FEATURES == ["receiver_avg_out_amount"]
    assert "receiver_avg_out_amount" not in ALL_FEATURES

    cfg = json.loads(Path("configs/tgat.json").read_text(encoding="utf-8"))
    assert cfg["model"]["transaction_dim"] == 26
    assert cfg["model"]["event_dim"] == 26
    assert cfg["model"]["num_layers"] == 2
    assert cfg["model"]["max_history"] == 10
    assert cfg["data"]["timestamp_resolution_seconds"] == 1

    default_model = TGATTransactionModel()
    assert default_model.transaction_dim == 26
    assert default_model.event_dim == 26

    try:
        TemporalFeatureScaler(
            mean=np.zeros(26),
            scale=np.ones(26),
            features=tuple(reversed(ALL_FEATURES)),
        )
        raise AssertionError("Expected ValueError for misordered feature contract")
    except ValueError:
        pass


def test_optimized_train_epoch_and_predict_bit_exact_parity() -> None:
    """Verify optimized neighbor store and pre-sampled train_epoch match naive reference identically."""
    rng = np.random.default_rng(123)
    rows = []
    base_ts = pd.Timestamp("2022-09-01 00:00:00")
    accounts = ["A0", "A1", "A2", "A3", "A4", "A5"]
    # Multiple same-timestamp groups with bi-directional edges to stress deque tie-breaking
    for g in range(8):
        offset = g * 45
        for r in range(7):
            src = accounts[(g + r) % len(accounts)]
            dst = accounts[(g + r + 1 + (r % 2)) % len(accounts)]
            amt = float(rng.uniform(10.0, 500.0))
            rec = {
                "From Account": src,
                "To Account": dst,
                "Timestamp": (base_ts + pd.Timedelta(seconds=offset)).strftime("%Y/%m/%d %H:%M:%S"),
                "Is Laundering": 1 if (g % 2 == 0 and r == 2) else 0,
            }
            for idx_f, name in enumerate(ALL_FEATURES):
                rec[name] = float(amt * 0.01 + idx_f * 0.1 + rng.normal(0.0, 0.25))
            rows.append(rec)

    full_df = pd.DataFrame(rows)
    full_df["_timestamp"] = to_unix_seconds(pd.to_datetime(full_df["Timestamp"], format="mixed"))
    train_df = full_df.iloc[:35].reset_index(drop=True)
    val_df = full_df.iloc[35:].reset_index(drop=True)
    scaler = TemporalFeatureScaler.fit(train_df)
    train_df = scaler.transform_frame(train_df)
    val_df = scaler.transform_frame(val_df)

    def ref_train_and_eval(use_neighbors: bool, dev: torch.device):
        seed_everything(77)
        model = TGATTransactionModel(dropout=0.0, use_time_encoding=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = torch.nn.BCEWithLogitsLoss()
        store = EventFeatureStore(max_history=4)

        cols = list(train_df.columns)
        idx_map = {n: i for i, n in enumerate(cols)}
        store_indices = (idx_map["From Account"], idx_map["To Account"], *(idx_map[c] for c in ALL_FEATURES))

        model.train()
        total_loss = 0.0
        used = pos = neg = n_batches = 0
        for _, grp in train_df.groupby("_timestamp", sort=False):
            for start in range(0, len(grp), 3):
                n_batches += 1
                batch = build_batch(grp.iloc[start : start + 3], store, neighbor_k=4, use_neighbors=use_neighbors)
                labels = batch.labels
                p_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
                n_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
                k_neg = min(len(n_idx), max(1, len(p_idx) * 2))
                if k_neg < len(n_idx):
                    n_idx = n_idx[torch.randperm(len(n_idx))[:k_neg]]
                keep = torch.cat([p_idx, n_idx])
                if len(keep) == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                logits, _ = model(
                    batch.transaction_features[keep].to(dev),
                    batch.sender_features[keep].to(dev),
                    batch.sender_delta_seconds[keep].to(dev),
                    batch.receiver_features[keep].to(dev),
                    batch.receiver_delta_seconds[keep].to(dev),
                )
                y = labels[keep].to(dev)
                loss = crit(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += float(loss.item()) * len(keep)
                used += len(keep)
                pos += int((y == 1).sum())
                neg += int((y == 0).sum())
            for row in grp.itertuples(index=False, name=None):
                store.add_row(row, int(row[idx_map["_timestamp"]]), feature_indices=store_indices)

        eval_store = EventFeatureStore(max_history=4)
        for row in train_df.itertuples(index=False, name=None):
            eval_store.add_row(row, int(row[idx_map["_timestamp"]]), feature_indices=store_indices)
        model.eval()
        val_scores = []
        with torch.no_grad():
            for _, grp in val_df.groupby("_timestamp", sort=False):
                for start in range(0, len(grp), 3):
                    vb = build_batch(grp.iloc[start : start + 3], eval_store, neighbor_k=4, use_neighbors=use_neighbors)
                    vlogits, _ = model(
                        vb.transaction_features.to(dev),
                        vb.sender_features.to(dev),
                        vb.sender_delta_seconds.to(dev),
                        vb.receiver_features.to(dev),
                        vb.receiver_delta_seconds.to(dev),
                    )
                    val_scores.extend(torch.sigmoid(vlogits).cpu().numpy().tolist())
                for row in grp.itertuples(index=False, name=None):
                    eval_store.add_row(row, int(row[idx_map["_timestamp"]]), feature_indices=store_indices)
        return total_loss / used, pos, neg, n_batches, {k: v.cpu() for k, v in model.state_dict().items()}, np.asarray(val_scores, dtype=np.float64)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        for use_nb in (True, False):
            ref_loss, ref_pos, ref_neg, ref_nb, ref_sd, ref_scores = ref_train_and_eval(use_nb, dev)

            seed_everything(77)
            fast_model = TGATTransactionModel(dropout=0.0, use_time_encoding=True).to(dev)
            fast_opt = torch.optim.AdamW(fast_model.parameters(), lr=1e-3, weight_decay=1e-4)
            fast_crit = torch.nn.BCEWithLogitsLoss()
            fast_store = EventFeatureStore(max_history=4)
            fast_loss, fast_pos, fast_neg, fast_nb = train_epoch(
                fast_model, train_df, fast_store, fast_opt, fast_crit,
                batch_size=3, neighbor_k=4, negative_ratio=2, device=dev,
                active_features=list(ALL_FEATURES), use_neighbors=use_nb,
            )
            fast_eval_store = EventFeatureStore(max_history=4)
            populate_history(train_df, fast_eval_store, batch_size=3, active_features=list(ALL_FEATURES), use_neighbors=use_nb)
            val_lbl, fast_scores, val_ts = predict(
                fast_model, val_df, fast_eval_store, batch_size=3, neighbor_k=4,
                device=dev, active_features=list(ALL_FEATURES), use_neighbors=use_nb,
            )

            assert abs(fast_loss - ref_loss) < 1e-12, f"{dev}, {use_nb}: {fast_loss} vs {ref_loss}"
            assert (fast_pos, fast_neg, fast_nb) == (ref_pos, ref_neg, ref_nb)
            for k, v in fast_model.state_dict().items():
                assert torch.equal(v.cpu(), ref_sd[k]), f"Mismatch weight {k} on {dev} use_nb={use_nb}"
            assert np.array_equal(fast_scores, ref_scores), f"Score mismatch on {dev} use_nb={use_nb}"
            assert np.array_equal(val_lbl, val_df[TARGET].to_numpy(dtype=np.int64))
            assert np.array_equal(val_ts, val_df["_timestamp"].to_numpy(dtype=np.int64))


if __name__ == "__main__":
    test_pandas_microsecond_quantization_repro_and_to_unix_seconds_fix()
    test_temporal_dataset_one_second_time_since_last_and_chunk_boundary()
    test_build_temporal_features_matches_one_second_unix()
    test_time_encoder_distinguishes_one_second_and_minute_deltas()
    test_event_store_and_make_batches_causality_and_reference_parity()
    test_feature_contract_26_synchronized()
    test_optimized_train_epoch_and_predict_bit_exact_parity()
    print("ALL RESEARCH VALIDATION TESTS PASSED")
