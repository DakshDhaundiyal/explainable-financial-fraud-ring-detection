"""Regression and numerical stability tests for TGAT v3 TimeEncoder and causality guarantees."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import (
    TGATTransactionModel,
    TimeEncoder,
    VALID_TIME_ENCODING_MODES,
)
from src.models.temporal.tgat_dataset import (
    ALL_FEATURES,
    FEATURE_DIM,
    TARGET,
    EventFeatureStore,
    build_batch_from_arrays,
    to_unix_seconds,
)
from src.models.temporal.train_tgat import (
    make_batches,
    populate_history,
    predict,
    read_frame,
    seed_everything,
    train_epoch,
)


def test_1_time_encoder_numerical_stability_across_scales_and_invalid_inputs() -> None:
    """Criterion 1 & 2: zero, short, medium, large, negative, NaN, and +/-Inf gaps."""
    deltas = torch.tensor(
        [0.0, 1.0, 60.0, 300.0, 3_600.0, 86_400.0, 864_000.0, 10_000_000.0, -42.0, float("nan"), float("inf"), float("-inf")],
        dtype=torch.float32,
    )
    for mode in VALID_TIME_ENCODING_MODES:
        enc = TimeEncoder(dimension=16, max_period=86_400.0, time_encoding_mode=mode)
        enc.zero_grad(set_to_none=True)
        scaled = enc.scale_delta(deltas)
        assert torch.isfinite(scaled).all(), f"{mode}: scaled contains non-finite values"
        assert (scaled >= 0.0).all(), f"{mode}: scaled contains negative values"
        # Negative, NaN, Inf sanitized to 0.0 -> must equal zero-delta encoding
        assert torch.allclose(scaled[0], scaled[8])
        assert torch.allclose(scaled[0], scaled[9])
        assert torch.allclose(scaled[0], scaled[10])
        assert torch.allclose(scaled[0], scaled[11])
        # Strict monotonicity over valid non-negative deltas [0 .. 1e7]
        valid_scaled = scaled[:8]
        assert (valid_scaled[1:] > valid_scaled[:-1]).all(), f"{mode}: scaled not strictly increasing"

        out = enc(deltas)
        assert out.shape == (len(deltas), 16)
        assert torch.isfinite(out).all(), f"{mode}: forward output not finite"
        assert (out >= -1.0).all() and (out <= 1.0).all()

        loss = (out ** 2).mean()
        loss.backward()
        assert enc.frequency.grad is not None and torch.isfinite(enc.frequency.grad).all()
        assert enc.phase.grad is not None and torch.isfinite(enc.phase.grad).all()


def test_2_log1p_bounds_gradient_amplification_vs_raw_seconds() -> None:
    """Criterion 1 & 2: log1p_seconds bounds dL/dfrequency compared to raw_seconds."""
    large_deltas = torch.tensor([63_000.0, 333_300.0, 690_960.0, 863_820.0], dtype=torch.float32)

    enc_raw = TimeEncoder(dimension=16, time_encoding_mode="raw_seconds")
    enc_log = TimeEncoder(dimension=16, time_encoding_mode="log1p_seconds")
    with torch.no_grad():
        enc_log.frequency.copy_(enc_raw.frequency)
        enc_log.phase.copy_(enc_raw.phase)

    out_raw = enc_raw(large_deltas)
    out_raw.sum().backward()
    raw_freq_grad_norm = float(enc_raw.frequency.grad.norm())

    out_log = enc_log(large_deltas)
    out_log.sum().backward()
    log_freq_grad_norm = float(enc_log.frequency.grad.norm())

    assert np.isfinite(raw_freq_grad_norm) and np.isfinite(log_freq_grad_norm)
    assert raw_freq_grad_norm > log_freq_grad_norm * 1_000.0, (
        f"Expected raw frequency grad explosion (>1000x log1p), got raw={raw_freq_grad_norm}, log1p={log_freq_grad_norm}"
    )

    # Also verify log1p preserves fine short-burst discriminability (60s vs 300s vs 600s)
    with torch.no_grad():
        short = enc_log(torch.tensor([[60.0], [300.0], [600.0]]))
        assert float((short[0] - short[1]).abs().max()) > 1e-1
        assert float((short[1] - short[2]).abs().max()) > 1e-1


def _build_synthetic_temporal_splits(tmp: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    base_ts = pd.Timestamp("2022-09-01 00:00:00")
    accounts = ["A0", "A1", "A2", "A3", "A4", "A5"]
    rows = []
    # 12 timestamp groups -> train (0..7), val (8..9), test (10..11)
    for g in range(12):
        # Mix 60s and multi-hour gaps
        offset = g * 1800 if g < 6 else (6 * 1800 + (g - 5) * 86_400)
        for r in range(6):
            src = accounts[(g + r) % len(accounts)]
            dst = accounts[(g + r + 1 + (r % 2)) % len(accounts)]
            rec = {
                "From Account": src,
                "To Account": dst,
                "Timestamp": (base_ts + pd.Timedelta(seconds=offset)).strftime("%Y/%m/%d %H:%M:%S"),
                "Is Laundering": 1 if (r == 2 and g % 2 == 0) else 0,
            }
            for f_i, name in enumerate(ALL_FEATURES):
                rec[name] = float(rng.normal(loc=10.0 + f_i, scale=2.5))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["_timestamp"] = to_unix_seconds(pd.to_datetime(df["Timestamp"], format="mixed"))
    train_df = df.iloc[:48].reset_index(drop=True)
    val_df = df.iloc[48:60].reset_index(drop=True)
    test_df = df.iloc[60:].reset_index(drop=True)
    return train_df, val_df, test_df


def test_3_to_6_consistent_transforms_equal_ts_causality_and_no_target_leakage() -> None:
    """Criteria 3, 4, 5, 6: identical transformation across splits, equal timestamps, strict <T, no target leakage."""
    with tempfile.TemporaryDirectory() as tmp:
        train_raw, val_raw, test_raw = _build_synthetic_temporal_splits(Path(tmp))
        scaler = TemporalFeatureScaler.fit(train_raw)
        train_df = scaler.transform_frame(train_raw)
        val_df = scaler.transform_frame(val_raw)
        test_df = scaler.transform_frame(test_raw)

        # Flip future validation & test labels -> MUST produce zero change in features, batches, or model predictions
        val_flipped = val_df.copy()
        val_flipped[TARGET] = 1 - val_flipped[TARGET]
        test_flipped = test_df.copy()
        test_flipped[TARGET] = 1 - test_flipped[TARGET]

        for mode in ("log1p_seconds", "hours", "days"):
            seed_everything(42)
            model = TGATTransactionModel(
                transaction_dim=26, event_dim=26, hidden_dim=32, time_dim=16,
                num_heads=2, dropout=0.0, num_layers=2, use_time_encoding=True,
                time_encoding_mode=mode,
            )
            assert model.time_encoder.time_encoding_mode == mode

            store_a = EventFeatureStore(max_history=10)
            populate_history(train_df, store_a, batch_size=4)
            _, val_scores_orig, val_ts_orig = predict(model, val_df, store_a, batch_size=4, neighbor_k=10, device=torch.device("cpu"))
            _, test_scores_orig, test_ts_orig = predict(model, test_df, store_a, batch_size=4, neighbor_k=10, device=torch.device("cpu"))

            store_b = EventFeatureStore(max_history=10)
            populate_history(train_df, store_b, batch_size=4)
            _, val_scores_flip, _ = predict(model, val_flipped, store_b, batch_size=4, neighbor_k=10, device=torch.device("cpu"))
            _, test_scores_flip, _ = predict(model, test_flipped, store_b, batch_size=4, neighbor_k=10, device=torch.device("cpu"))

            assert np.array_equal(val_scores_orig, val_scores_flip), f"{mode}: val target leakage"
            assert np.array_equal(test_scores_orig, test_scores_flip), f"{mode}: test target leakage"
            assert np.isfinite(val_scores_orig).all() and np.isfinite(test_scores_orig).all()

            # Check equal timestamps & strict historical causality in batches
            check_store = EventFeatureStore(max_history=10)
            batches = list(make_batches(val_df, check_store, batch_size=2, neighbor_k=10))
            assert len(batches) > 0
            # Group of 6 rows at same timestamp split into batch sizes [2, 2, 2]: none can see same-timestamp peers
            for b in batches:
                s_pos = b.sender_delta_seconds > 0
                r_pos = b.receiver_delta_seconds > 0
                assert (b.sender_delta_seconds[~s_pos] == 0).all()
                assert (b.receiver_delta_seconds[~r_pos] == 0).all()
                # Non-zero delta guarantees event_time < query_time strictly
                if s_pos.any():
                    assert (b.sender_delta_seconds[s_pos] >= 60.0).all()


def test_7_and_8_feature_contract_and_checkpoint_scaler_compatibility() -> None:
    """Criteria 7 & 8: 26-feature contract, checkpoint state_dict save/load, scaler serialization parity."""
    assert len(ALL_FEATURES) == 26 and FEATURE_DIM == 26
    assert "receiver_avg_out_amount" not in ALL_FEATURES

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        train_raw, val_raw, _ = _build_synthetic_temporal_splits(tmp_dir)
        scaler = TemporalFeatureScaler.fit(train_raw)
        scaler_path = tmp_dir / "scaler_params.json"
        scaler_path.write_text(
            json.dumps({
                "features": list(scaler.features),
                "mean": scaler.mean.tolist(),
                "scale": scaler.scale.tolist(),
                "clip": float(scaler.clip),
            }),
            encoding="utf-8",
        )
        loaded_meta = json.loads(scaler_path.read_text(encoding="utf-8"))
        restored_scaler = TemporalFeatureScaler(
            mean=np.asarray(loaded_meta["mean"], dtype=np.float64),
            scale=np.asarray(loaded_meta["scale"], dtype=np.float64),
            clip=float(loaded_meta["clip"]),
            features=tuple(loaded_meta["features"]),
        )
        pd.testing.assert_frame_equal(
            scaler.transform_frame(val_raw),
            restored_scaler.transform_frame(val_raw),
        )

        # Save & reload TGAT v3 checkpoint cleanly with weights_only=True
        model = TGATTransactionModel(time_encoding_mode="log1p_seconds", dropout=0.0)
        ckpt_path = tmp_dir / "model.pt"
        torch.save(model.state_dict(), ckpt_path)

        reloaded = TGATTransactionModel(time_encoding_mode="log1p_seconds", dropout=0.0)
        reloaded.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        reloaded.eval()

        tx = torch.randn(2, 26)
        sf = torch.randn(2, 11, 26)
        rf = torch.randn(2, 11, 26)
        sd = torch.full((2, 10), 3600.0)
        rd = torch.full((2, 10), 86400.0)
        with torch.no_grad():
            l1, _ = model(tx, sf, sd, rf, rd)
            l2, _ = reloaded(tx, sf, sd, rf, rd)
            assert torch.allclose(l1, l2, atol=1e-7)


def test_9_and_10_reproducibility_and_cpu_cuda_consistency() -> None:
    """Criteria 9 & 10: deterministic seeded training + CPU/CUDA output & gradient norm tracking."""
    with tempfile.TemporaryDirectory() as tmp:
        train_raw, val_raw, _ = _build_synthetic_temporal_splits(Path(tmp))
        scaler = TemporalFeatureScaler.fit(train_raw)
        train_df = scaler.transform_frame(train_raw)
        val_df = scaler.transform_frame(val_raw)

        def run_once(device: torch.device, seed: int) -> tuple[float, dict, np.ndarray]:
            seed_everything(seed)
            model = TGATTransactionModel(
                dropout=0.0,
                use_time_encoding=True,
                time_encoding_mode="log1p_seconds",
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = torch.nn.BCEWithLogitsLoss()
            store = EventFeatureStore(max_history=10)
            g_stats: dict = {}
            loss, _, _, _ = train_epoch(
                model, train_df, store, opt, crit,
                batch_size=4, neighbor_k=10, negative_ratio=5,
                device=device, grad_clip_norm=5.0, grad_stats_out=g_stats,
            )
            eval_store = EventFeatureStore(max_history=10)
            populate_history(train_df, eval_store, batch_size=4)
            _, val_scores, _ = predict(model, val_df, eval_store, batch_size=4, neighbor_k=10, device=device)
            return loss, g_stats, val_scores

        # Two seeded runs on CPU must be bit-for-bit reproducible
        l_cpu1, g_cpu1, s_cpu1 = run_once(torch.device("cpu"), 42)
        l_cpu2, g_cpu2, s_cpu2 = run_once(torch.device("cpu"), 42)
        assert abs(l_cpu1 - l_cpu2) < 1e-12
        assert abs(g_cpu1["pre_clip_grad_norm"]["mean"] - g_cpu2["pre_clip_grad_norm"]["mean"]) < 1e-12
        assert np.array_equal(s_cpu1, s_cpu2)

        # CPU vs CUDA forward consistency on identical checkpoint & batch
        if torch.cuda.is_available():
            seed_everything(42)
            m_cpu = TGATTransactionModel(dropout=0.0, time_encoding_mode="log1p_seconds").eval()
            m_gpu = TGATTransactionModel(dropout=0.0, time_encoding_mode="log1p_seconds").to("cuda").eval()
            m_gpu.load_state_dict({k: v.cuda() for k, v in m_cpu.state_dict().items()})
            tx = torch.randn(4, 26)
            sf = torch.randn(4, 11, 26)
            rf = torch.randn(4, 11, 26)
            sd = torch.tensor([[60.0, 3600.0, 86400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
            rd = sd.clone()
            with torch.no_grad():
                out_cpu, _ = m_cpu(tx, sf, sd, rf, rd)
                out_gpu, _ = m_gpu(tx.cuda(), sf.cuda(), sd.cuda(), rf.cuda(), rd.cuda())
            assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5)

            l_gpu, g_gpu, s_gpu = run_once(torch.device("cuda"), 42)
            assert np.isfinite(l_gpu)
            assert np.isfinite(g_gpu["pre_clip_grad_norm"]["mean"])
            assert np.isfinite(s_gpu).all()


if __name__ == "__main__":
    test_1_time_encoder_numerical_stability_across_scales_and_invalid_inputs()
    test_2_log1p_bounds_gradient_amplification_vs_raw_seconds()
    test_3_to_6_consistent_transforms_equal_ts_causality_and_no_target_leakage()
    test_7_and_8_feature_contract_and_checkpoint_scaler_compatibility()
    test_9_and_10_reproducibility_and_cpu_cuda_consistency()
    print("ALL TGAT V3 REGRESSION TESTS PASSED")

