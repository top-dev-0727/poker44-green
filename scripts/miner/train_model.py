"""Train a Poker44 bot-detection model from the public benchmark API.

Example
-------
    python scripts/miner/train_model.py

Environment variables
---------------------
    POKER44_BENCHMARK_BASE_URL    Base URL for the benchmark API (default: https://api.poker44.net/api/v1/benchmark)
    POKER44_BENCHMARK_SOURCE_DATE  Date to download (default: latest)
    POKER44_BENCHMARK_CHUNK_LIMIT  Chunks per page (default: 24)
    POKER44_MODEL_OUTPUT_DIR       Where to save model.pkl and metadata.json (default: ./models)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Make the repo root importable regardless of where the script is invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poker44.miner.features import extract_chunk_features, feature_names
from poker44.miner.model import BotDetectionModel


DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> Tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.any(y_true == 1):
        return float(average_precision_score(y_true, y_score))
    return 0.0


def fetch_benchmark_status(base_url: str) -> Dict[str, Any]:
    response = requests.get(base_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def fetch_benchmark_chunks(
    base_url: str,
    source_date: str,
    *,
    limit: int = 24,
    max_chunks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Download all benchmark chunks for a source date."""
    chunks: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        params: Dict[str, Any] = {"sourceDate": source_date, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(f"{base_url}/chunks", params=params, timeout=120)
        response.raise_for_status()
        data = response.json().get("data", {})

        batch = data.get("chunks", [])
        if not batch:
            break

        chunks.extend(batch)
        page += 1
        print(
            f"Downloaded page {page}: {len(batch)} chunks "
            f"(total {len(chunks)})"
        )

        if max_chunks is not None and len(chunks) >= max_chunks:
            chunks = chunks[:max_chunks]
            break

        cursor = data.get("nextCursor")
        if not cursor:
            break

    return chunks


def build_dataset(chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and label vector from benchmark chunks."""
    rows: List[List[float]] = []
    labels: List[int] = []

    for chunk_wrapper in chunks:
        chunk_groups = chunk_wrapper.get("chunks") or []
        ground_truth = chunk_wrapper.get("groundTruth") or chunk_wrapper.get("groundTruthLabels")

        if not chunk_groups or not ground_truth:
            continue

        # Normalize string labels to integers
        if isinstance(ground_truth[0], str):
            label_map = {"human": 0, "bot": 1}
            numeric_labels = [label_map.get(str(l).lower(), 0) for l in ground_truth]
        else:
            numeric_labels = [int(l) for l in ground_truth]

        if len(chunk_groups) != len(numeric_labels):
            print(
                f"Skipping chunk {chunk_wrapper.get('chunkId')}: "
                f"mismatch between groups ({len(chunk_groups)}) and labels ({len(numeric_labels)})"
            )
            continue

        for group, label in zip(chunk_groups, numeric_labels):
            if not isinstance(group, list):
                continue
            feats = extract_chunk_features(group)
            rows.append([float(feats[name]) for name in feature_names()])
            labels.append(label)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    ap = _average_precision(y_true, y_score)
    bot_recall, fpr = _recall_at_fpr(y_score, y_true, max_fpr=0.05)
    base_score = 0.75 * ap + 0.25 * bot_recall
    return {
        "ap_score": ap,
        "bot_recall_at_5pct_fpr": bot_recall,
        "fpr_at_recall": fpr,
        "base_score": base_score,
        "auc": _auc_roc(y_true, y_score),
    }


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.15,
    random_state: int = 44,
) -> BotDetectionModel:
    """Train and calibrate a HistGradientBoostingClassifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training samples: {len(y_train)} (bots: {np.sum(y_train)}, humans: {len(y_train) - np.sum(y_train)})")
    print(f"Test samples: {len(y_test)} (bots: {np.sum(y_test)}, humans: {len(y_test) - np.sum(y_test)})")

    # Compute sample weights to mitigate class imbalance in the benchmark.
    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    pos_weight = _safe_div(len(y_train), 2.0 * n_pos, 1.0)
    neg_weight = _safe_div(len(y_train), 2.0 * n_neg, 1.0)
    sample_weight = np.where(y_train == 1, pos_weight, neg_weight)

    base_estimator = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_depth=7,
        min_samples_leaf=15,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=0,
    )

    # Calibrate probabilities on held-out folds so that ranking and
    # threshold-based recall@FPR are both well-behaved.
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method="isotonic",
        cv=5,
        n_jobs=-1,
    )
    calibrated.fit(X_train, y_train, sample_weight=sample_weight)

    train_score = calibrated.predict_proba(X_train)[:, 1]
    test_score = calibrated.predict_proba(X_test)[:, 1]

    print("Train metrics:", evaluate(y_train, train_score))
    print("Test metrics:", evaluate(y_test, test_score))

    # Refit on the full dataset for deployment with balanced sample weights.
    n_pos_full = int(np.sum(y))
    n_neg_full = int(len(y) - n_pos_full)
    pos_weight_full = _safe_div(len(y), 2.0 * n_pos_full, 1.0)
    neg_weight_full = _safe_div(len(y), 2.0 * n_neg_full, 1.0)
    sample_weight_full = np.where(y == 1, pos_weight_full, neg_weight_full)
    calibrated.fit(X, y, sample_weight=sample_weight_full)

    return BotDetectionModel(
        model=calibrated,
        feature_names=feature_names(),
        calibration_info={"method": "isotonic", "cv": 5},
    )


def main() -> int:
    base_url = os.getenv("POKER44_BENCHMARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    output_dir = Path(os.getenv("POKER44_MODEL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    chunk_limit = int(os.getenv("POKER44_BENCHMARK_CHUNK_LIMIT", "24"))
    max_chunks_env = os.getenv("POKER44_BENCHMARK_MAX_CHUNKS")
    max_chunks = int(max_chunks_env) if max_chunks_env else None

    print(f"Benchmark API: {base_url}")
    status = fetch_benchmark_status(base_url)
    source_date = os.getenv("POKER44_BENCHMARK_SOURCE_DATE") or status.get("latestSourceDate")
    print(f"Using source date: {source_date}")

    chunks = fetch_benchmark_chunks(
        base_url,
        source_date=source_date,
        limit=chunk_limit,
        max_chunks=max_chunks,
    )
    print(f"Total chunks downloaded: {len(chunks)}")

    if not chunks:
        print("No benchmark chunks available; cannot train.")
        return 1

    X, y = build_dataset(chunks)
    print(f"Dataset shape: {X.shape}, positive rate: {np.mean(y):.4f}")

    if X.shape[0] < 100:
        print("Insufficient data for training.")
        return 1

    model = train_model(X, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = model.save(
        model_path=output_dir / "bot_detector.pkl",
        metadata_path=output_dir / "bot_detector_metadata.json",
        extra_metadata={
            "training_source_date": source_date,
            "training_samples": int(X.shape[0]),
            "training_bots": int(np.sum(y)),
            "training_humans": int(np.sum(1 - y)),
            "training_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_api": base_url,
        },
    )

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
