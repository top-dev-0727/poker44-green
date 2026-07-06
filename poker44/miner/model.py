"""Trained bot-detection model wrapper for the Poker44 miner.

The wrapper handles model persistence, feature alignment, and calibrated
probability output.  It is intentionally lightweight so that inference stays
fast inside the Bittensor axon forward path.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from poker44.miner.features import extract_chunk_features, feature_names


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "bot_detector.pkl"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[2] / "models" / "bot_detector_metadata.json"


class BotDetectionModel:
    """Loadable GBDT-based bot-risk scorer with feature-name alignment."""

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        calibration_info: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.feature_names = tuple(feature_names)
        self.calibration_info = calibration_info or {}

    @classmethod
    def load(
        cls,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
    ) -> "BotDetectionModel":
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as handle:
            model = pickle.load(handle)

        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        return cls(
            model=model,
            feature_names=metadata.get("feature_names", list(feature_names())),
            calibration_info=metadata.get("calibration", {}),
        )

    def _build_matrix(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        rows: List[List[float]] = []
        for chunk in chunks:
            feats = extract_chunk_features(chunk)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        return np.asarray(rows, dtype=float)

    def predict_proba(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        """Return a calibrated bot probability for each chunk."""
        if not chunks:
            return np.asarray([], dtype=float)

        X = self._build_matrix(chunks)

        # Some scikit-learn estimators raise warnings on all-zero rows; suppress
        # them because the feature extractor already handles empty chunks safely.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probas = self.model.predict_proba(X)[:, 1]

        return np.asarray(probas, dtype=float)

    def predict(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        """Return a boolean bot prediction for each chunk."""
        return self.predict_proba(chunks) >= 0.5

    def save(
        self,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata: Dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "calibration": self.calibration_info,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        return model_path, metadata_path
