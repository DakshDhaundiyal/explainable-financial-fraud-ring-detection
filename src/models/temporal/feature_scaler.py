"""Train-only feature scaling for temporal neural models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.temporal.tgat_dataset import ALL_FEATURES


@dataclass
class TemporalFeatureScaler:
    """Standardize temporal TGAT features using training data only."""

    mean: np.ndarray
    scale: np.ndarray
    clip: float = 10.0
    features: tuple[str, ...] = field(default_factory=lambda: tuple(ALL_FEATURES))

    def __post_init__(self) -> None:
        if len(self.features) != len(ALL_FEATURES) or list(self.features) != ALL_FEATURES:
            raise ValueError(
                f"Scaler feature contract mismatch: expected {len(ALL_FEATURES)} ALL_FEATURES, "
                f"got {len(self.features)}"
            )
        if self.mean.shape != (len(ALL_FEATURES),) or self.scale.shape != (len(ALL_FEATURES),):
            raise ValueError("Scaler mean/scale shapes must match ALL_FEATURES length")

    @classmethod
    def fit(cls, frame: pd.DataFrame, clip: float = 10.0) -> "TemporalFeatureScaler":
        # Use pandas column-wise reductions instead of materializing the full
        # training matrix as float64. This matters for multi-million-row data.
        missing = [col for col in ALL_FEATURES if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing TGAT features for scaler fit: {missing}")
        values = frame[ALL_FEATURES]
        mean = values.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        scale = values.std(axis=0, ddof=0, skipna=True).to_numpy(dtype=np.float64)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean=mean, scale=scale, clip=clip, features=tuple(ALL_FEATURES))

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        transformed = (values - self.mean) / self.scale
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in ALL_FEATURES if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing TGAT features for scaler transform: {missing}")
        result = frame.copy()
        result.loc[:, ALL_FEATURES] = self.transform_array(
            result[ALL_FEATURES].to_numpy(dtype=np.float64)
        )
        return result
