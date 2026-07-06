"""Unit tests for the Poker44 miner feature extractor and model wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poker44.miner.features import extract_chunk_features, feature_names, feature_vector
from poker44.miner.model import BotDetectionModel


class MinerFeatureTests(unittest.TestCase):
    def _sample_hand(self, bot_like: bool = False) -> dict:
        if bot_like:
            actions = [
                {"action_type": "raise", "actor_seat": 1, "normalized_amount_bb": 2.5, "pot_before": 1.5, "pot_after": 4.0},
                {"action_type": "fold", "actor_seat": 2, "normalized_amount_bb": 0.0, "pot_before": 4.0, "pot_after": 4.0},
            ] * 5
        else:
            actions = [
                {"action_type": "call", "actor_seat": 1, "normalized_amount_bb": 1.0, "pot_before": 2.0, "pot_after": 3.0},
                {"action_type": "check", "actor_seat": 2, "normalized_amount_bb": 0.0, "pot_before": 3.0, "pot_after": 3.0},
                {"action_type": "fold", "actor_seat": 3, "normalized_amount_bb": 0.0, "pot_before": 3.0, "pot_after": 3.0},
            ] * 3
        return {
            "metadata": {"game_type": "holdem", "limit_type": "nl", "max_seats": 6, "hero_seat": 1},
            "players": [{"seat": 1, "starting_stack": 100.0}, {"seat": 2, "starting_stack": 100.0}],
            "streets": [{"street": "preflop"}, {"street": "flop"}],
            "actions": actions,
            "outcome": {"showdown": not bot_like},
        }

    def test_feature_vector_shape_and_finite(self) -> None:
        chunk = [self._sample_hand(bot_like=False), self._sample_hand(bot_like=True)]
        vec = feature_vector(chunk)
        self.assertEqual(vec.shape, (len(feature_names()),))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_empty_chunk_returns_zeros(self) -> None:
        feats = extract_chunk_features([])
        vec = feature_vector([])
        self.assertEqual(len(feats), len(feature_names()))
        self.assertTrue(np.allclose(vec, 0.0))

    def test_feature_names_are_stable(self) -> None:
        names_a = feature_names()
        names_b = feature_names()
        self.assertEqual(names_a, names_b)


class DummyClassifier:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - scores, scores])


class BotDetectionModelTests(unittest.TestCase):
    def test_predict_proba_returns_expected_shape(self) -> None:
        model = BotDetectionModel(
            model=DummyClassifier(),
            feature_names=feature_names(),
        )
        chunks = [
            [{"actions": [], "players": [], "streets": [], "outcome": {}, "metadata": {}}],
            [{"actions": [], "players": [], "streets": [], "outcome": {}, "metadata": {}}],
        ]
        probs = model.predict_proba(chunks)
        self.assertEqual(probs.shape, (2,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_save_and_load_roundtrip(self) -> None:
        model = BotDetectionModel(
            model=DummyClassifier(),
            feature_names=feature_names(),
            calibration_info={"method": "isotonic"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            meta_path = Path(tmp_dir) / "meta.json"
            model.save(model_path=model_path, metadata_path=meta_path, extra_metadata={"foo": "bar"})

            loaded = BotDetectionModel.load(model_path=model_path, metadata_path=meta_path)
            self.assertEqual(loaded.feature_names, model.feature_names)
            self.assertEqual(loaded.calibration_info, model.calibration_info)

            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["foo"], "bar")


if __name__ == "__main__":
    unittest.main()
