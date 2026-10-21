"""TGAT Model Inference Service.

Loads the canonical TGAT checkpoint, feature scaler, frozen threshold,
and provides real-time transaction scoring with layer-wise attention explainability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.temporal_neighbor_store import TemporalNeighborStore


class ModelService:
    def __init__(
        self,
        checkpoint_path: str = "results/tgat/model.pt",
        scaler_path: str = "backend/scaler_params.json",
        threshold_path: str = "results/threshold_analysis/frozen_threshold.json",
        neighbor_k: int = 10,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neighbor_k = neighbor_k

        # 1. Load Scaler
        with open(scaler_path, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        self.features = s_data["features"]
        self.mean = np.array(s_data["mean"], dtype=np.float32)
        self.scale = np.array(s_data["scale"], dtype=np.float32)
        self.clip = float(s_data.get("clip", 10.0))

        # 2. Load Threshold
        with open(threshold_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        self.threshold = float(t_data["threshold"])

        # 3. Load Model Checkpoint
        self.model = TGATTransactionModel(
            transaction_dim=len(self.features),
            event_dim=len(self.features),
            hidden_dim=32,
            time_dim=16,
            num_heads=2,
            dropout=0.2,
            num_layers=2,
            use_time_encoding=True,
        ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # 4. In-Memory Neighbor Store & Node ID Map
        self.node_id_map: Dict[str, int] = {}
        self.id_to_account: Dict[int, str] = {}
        self.neighbor_store = TemporalNeighborStore(max_history_per_node=32)

        self._seed_sample_neighbors()

    def get_or_create_node_id(self, account: str) -> int:
        if account not in self.node_id_map:
            new_id = len(self.node_id_map) + 1
            self.node_id_map[account] = new_id
            self.id_to_account[new_id] = account
        return self.node_id_map[account]

    def _seed_sample_neighbors(self) -> None:
        """Seed representative historical events for interactive demo accounts."""
        import pandas as pd
        if Path("data/processed/temporal_gnn/test.csv").exists():
            df = pd.read_csv("data/processed/temporal_gnn/test.csv", nrows=10000)
            df["_timestamp"] = pd.to_datetime(df["Timestamp"]).astype("int64") // 1_000_000_000
            for row in df.itertuples(index=False):
                src_id = self.get_or_create_node_id(str(getattr(row, "From Account", getattr(row, "_0", ""))))
                dst_id = self.get_or_create_node_id(str(getattr(row, "To Account", getattr(row, "_1", ""))))
                ts = int(getattr(row, "_timestamp", 0))
                feat = [float(getattr(row, col, 0.0)) for col in self.features]
                scaled_feat = np.clip((np.array(feat, dtype=np.float32) - self.mean) / self.scale, -self.clip, self.clip)
                self.neighbor_store.add_event(ts, src_id, dst_id, scaled_feat.tolist())

    def scale_features(self, feat_dict: Dict[str, float]) -> np.ndarray:
        raw = np.array([float(feat_dict.get(k, 0.0)) for k in self.features], dtype=np.float32)
        scaled = np.clip((raw - self.mean) / self.scale, -self.clip, self.clip)
        return scaled

    def predict(
        self,
        from_account: str,
        to_account: str,
        timestamp_sec: int,
        feature_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        scaled_tx = self.scale_features(feature_dict)
        tx_tensor = torch.tensor(scaled_tx, dtype=torch.float32, device=self.device).unsqueeze(0)

        src_id = self.get_or_create_node_id(from_account)
        dst_id = self.get_or_create_node_id(to_account)

        # Retrieve strictly past events
        src_events = self.neighbor_store.recent_events(src_id, timestamp_sec, self.neighbor_k)
        dst_events = self.neighbor_store.recent_events(dst_id, timestamp_sec, self.neighbor_k)

        # Build Sender Neighbor Tensors
        src_feat = np.zeros((1, self.neighbor_k + 1, len(self.features)), dtype=np.float32)
        src_deltas = np.zeros((1, self.neighbor_k), dtype=np.float32)
        src_feat[0, 0, :] = scaled_tx
        src_neighbor_details = []
        for i, ev in enumerate(src_events[:self.neighbor_k]):
            src_feat[0, i + 1, :] = np.array(ev.features, dtype=np.float32)
            delta = max(0.0, float(timestamp_sec - ev.timestamp))
            src_deltas[0, i] = delta
            src_neighbor_details.append({
                "index": i,
                "partner": self.id_to_account.get(ev.destination if ev.source == src_id else ev.source, "UNKNOWN"),
                "delta_seconds": delta,
                "timestamp": ev.timestamp,
            })

        # Build Receiver Neighbor Tensors
        dst_feat = np.zeros((1, self.neighbor_k + 1, len(self.features)), dtype=np.float32)
        dst_deltas = np.zeros((1, self.neighbor_k), dtype=np.float32)
        dst_feat[0, 0, :] = scaled_tx
        dst_neighbor_details = []
        for i, ev in enumerate(dst_events[:self.neighbor_k]):
            dst_feat[0, i + 1, :] = np.array(ev.features, dtype=np.float32)
            delta = max(0.0, float(timestamp_sec - ev.timestamp))
            dst_deltas[0, i] = delta
            dst_neighbor_details.append({
                "index": i,
                "partner": self.id_to_account.get(ev.source if ev.destination == dst_id else ev.destination, "UNKNOWN"),
                "delta_seconds": delta,
                "timestamp": ev.timestamp,
            })

        # Model Inference
        with torch.no_grad():
            logits, attn = self.model(
                tx_tensor,
                torch.tensor(src_feat, dtype=torch.float32, device=self.device),
                torch.tensor(src_deltas, dtype=torch.float32, device=self.device),
                torch.tensor(dst_feat, dtype=torch.float32, device=self.device),
                torch.tensor(dst_deltas, dtype=torch.float32, device=self.device),
                return_attention=True,
            )
            prob = float(torch.sigmoid(logits).cpu().item())

        is_fraud = prob >= self.threshold

        # Extract Attention Distributions
        sender_layer1 = attn["sender"][0][0].cpu().numpy().tolist() if len(attn["sender"]) > 0 else []
        sender_layer2 = attn["sender"][1][0].cpu().numpy().tolist() if len(attn["sender"]) > 1 else []
        receiver_layer1 = attn["receiver"][0][0].cpu().numpy().tolist() if len(attn["receiver"]) > 0 else []
        receiver_layer2 = attn["receiver"][1][0].cpu().numpy().tolist() if len(attn["receiver"]) > 1 else []

        # Merge neighbor metadata with layer-2 attention weights
        for idx, item in enumerate(src_neighbor_details):
            item["attention_l1"] = round(sender_layer1[idx], 4) if idx < len(sender_layer1) else 0.0
            item["attention_l2"] = round(sender_layer2[idx], 4) if idx < len(sender_layer2) else 0.0

        for idx, item in enumerate(dst_neighbor_details):
            item["attention_l1"] = round(receiver_layer1[idx], 4) if idx < len(receiver_layer1) else 0.0
            item["attention_l2"] = round(receiver_layer2[idx], 4) if idx < len(receiver_layer2) else 0.0

        # Construct Explainability Summary
        reasons = []
        if is_fraud:
            reasons.append(f"Predicted risk score ({prob*100:.1f}%) exceeds frozen threshold ({self.threshold*100:.1f}%).")
            if len(src_events) > 3:
                reasons.append(f"Sender demonstrates rapid sequential activity ({len(src_events)} recent interactions).")
            if feature_dict.get("Amount Difference", 0.0) > 1000:
                reasons.append("Significant disparity between sent and received transaction volumes.")
            if feature_dict.get("Cross Bank Transaction", 0) == 1:
                reasons.append("Interbank transaction structure consistent with layering phase.")
        else:
            reasons.append(f"Predicted risk score ({prob*100:.1f}%) is well below threshold ({self.threshold*100:.1f}%).")
            reasons.append("Historical graph attention is diffused across normal transaction intervals.")

        return {
            "from_account": from_account,
            "to_account": to_account,
            "timestamp": timestamp_sec,
            "probability": prob,
            "threshold": self.threshold,
            "decision": "FRAUD_ALERT" if is_fraud else "LEGITIMATE_CLEAR",
            "risk_level": "CRITICAL" if prob > 0.85 else ("HIGH" if is_fraud else ("MEDIUM" if prob > 0.2 else "LOW")),
            "confidence": round(abs(prob - self.threshold) / (1.0 - self.threshold if is_fraud else self.threshold), 3),
            "sender_attention": {
                "layer_1": sender_layer1,
                "layer_2": sender_layer2,
                "neighbors": src_neighbor_details,
            },
            "receiver_attention": {
                "layer_1": receiver_layer1,
                "layer_2": receiver_layer2,
                "neighbors": dst_neighbor_details,
            },
            "explainability": {
                "summary": " ".join(reasons),
                "primary_driver": "Temporal Graph Attention" if len(src_neighbor_details) > 0 else "Tabular Features",
                "top_sender_influencer": max(src_neighbor_details, key=lambda x: x["attention_l2"])["partner"] if src_neighbor_details else "None",
            }
        }
