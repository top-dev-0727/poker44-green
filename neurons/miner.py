"""Production Poker44 miner with trained GBDT bot detection.

The miner loads a calibrated classifier trained on the public Poker44 benchmark
API.  If the model artifact is not present, it falls back to an improved
behavioral heuristic so that the axon remains operational.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt
import numpy as np

from poker44.base.miner import BaseMinerNeuron
from poker44.miner.model import BotDetectionModel
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


# Repository identity for the manifest.  These must match the repository that
# actually hosts the served implementation.
REPO_URL = "https://github.com/top-dev-0727/poker44-green"
MODEL_NAME = "poker44-gbdt-bot-detector"
MODEL_VERSION = "1.0.0"


def _resolve_repo_commit(repo_root: Path) -> Optional[str]:
    """Return the current git commit hash, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        if len(commit) >= 7:
            return commit
    except Exception:
        pass
    return None


class Miner(BaseMinerNeuron):
    """Poker44 miner serving a trained bot-risk model with heuristic fallback."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 production miner starting")

        repo_root = Path(__file__).resolve().parents[1]
        self._load_model(repo_root)
        self.model_manifest = self._build_manifest(repo_root)
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _load_model(self, repo_root: Path) -> None:
        """Attempt to load the trained model; fall back to heuristic if missing."""
        model_path = repo_root / "models" / "bot_detector.pkl"
        try:
            self.detector = BotDetectionModel.load(model_path)
            bt.logging.info(f"Loaded trained model from {model_path}")
        except FileNotFoundError:
            bt.logging.warning(
                f"No trained model found at {model_path}; "
                "using improved heuristic fallback. Run scripts/miner/train_model.py to train."
            )
            self.detector = None
        except Exception as exc:
            bt.logging.error(f"Failed to load model: {exc}; using heuristic fallback.")
            self.detector = None

    def _build_manifest(self, repo_root: Path) -> dict:
        """Build a manifest that accurately describes the served implementation."""
        implementation_files = [
            repo_root / "neurons" / "miner.py",
            repo_root / "poker44" / "miner" / "__init__.py",
            repo_root / "poker44" / "miner" / "features.py",
            repo_root / "poker44" / "miner" / "model.py",
        ]

        if self.detector is not None:
            training_statement = (
                "Trained on public Poker44 benchmark releases served by "
                "https://api.poker44.net/api/v1/benchmark. The model is a "
                "calibrated histogram-based gradient boosting classifier "
                "trained on chunk-level behavioral features extracted from "
                "miner-visible poker hand payloads."
            )
            training_sources = ["https://api.poker44.net/api/v1/benchmark"]
            framework = "scikit-learn"
            inference_mode = "local"
            notes = (
                "Production GBDT miner for Poker44. Serves a calibrated "
                "classifier trained on the public benchmark API."
            )
        else:
            training_statement = (
                "Heuristic fallback miner. No training step. Uses only "
                "runtime chunk features when the trained model artifact is unavailable."
            )
            training_sources = ["none"]
            framework = "python-heuristic"
            inference_mode = "local"
            notes = (
                "Heuristic fallback mode. Run scripts/miner/train_model.py "
                "to generate the trained model artifact."
            )

        manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "framework": framework,
                "license": "MIT",
                "repo_url": REPO_URL,
                "notes": notes,
                "open_source": True,
                "inference_mode": inference_mode,
                "training_data_statement": training_statement,
                "training_data_sources": training_sources,
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data. "
                    "Training is performed exclusively on the public Poker44 benchmark API. "
                    "At inference time the miner receives only the miner-visible hand payloads "
                    "sent by validators and returns bot-risk scores; no ground-truth labels are used."
                ),
            },
        )

        # Ensure the manifest commit matches the code that is actually running.
        # The env var takes precedence so operators can pin a release commit.
        env_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
        manifest["repo_commit"] = env_commit or _resolve_repo_commit(repo_root) or manifest.get("repo_commit", "")
        return manifest

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one calibrated bot-risk score per chunk."""
        chunks = synapse.chunks or []

        if self.detector is not None and chunks:
            try:
                scores = self.detector.predict_proba(chunks).tolist()
                scores = [round(float(s), 6) for s in scores]
            except Exception as exc:
                bt.logging.warning(f"Model inference failed: {exc}; falling back to heuristic.")
                scores = [self._score_chunk_heuristic(chunk) for chunk in chunks]
        else:
            scores = [self._score_chunk_heuristic(chunk) for chunk in chunks]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_chunk_heuristic(cls, chunk: List[dict]) -> float:
        """Improved deterministic heuristic used when the model is unavailable."""
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand_heuristic(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        # Bots tend to be more regular across hands; boost score when hand-level
        # scores are tightly clustered.
        if len(hand_scores) > 1:
            std = float(np.std(hand_scores))
            regularity_bonus = 0.05 * (1.0 - cls._clamp01(std * 4.0))
            avg_score = cls._clamp01(avg_score + regularity_bonus)

        return round(avg_score, 6)

    @classmethod
    def _score_hand_heuristic(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(str(a.get("action_type")).lower() for a in actions)
        meaningful_actions = max(
            1,
            sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        bet_ratio = action_counts.get("bet", 0) / meaningful_actions
        street_depth = len(streets) / 4.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        # Action-amount regularity: bots often use round, predictable sizing.
        amounts = [float(a.get("normalized_amount_bb", 0.0) or 0.0) for a in actions]
        amount_regularity = 0.0
        if len(amounts) > 1 and np.mean(amounts) > 0:
            cv = float(np.std(amounts)) / float(np.mean(amounts))
            amount_regularity = 1.0 - cls._clamp01(cv / 2.0)

        score = 0.0
        score += 0.22 * street_depth
        score += 0.16 * showdown_flag
        score += 0.14 * cls._clamp01(call_ratio / 0.35)
        score += 0.10 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score += 0.10 * amount_regularity
        score += 0.08 * cls._clamp01((raise_ratio + bet_ratio) / 0.25)
        score -= 0.16 * cls._clamp01(fold_ratio / 0.55)

        return cls._clamp01(score)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 production miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
