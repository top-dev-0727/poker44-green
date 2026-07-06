"""Feature extraction for Poker44 bot-detection miner.

The functions here consume the miner-visible hand payload produced by
``poker44.validator.payload_view.prepare_hand_for_miner`` and emit a fixed-length
numeric feature vector per chunk.  The feature set is designed to capture
behavioral regularities that distinguish automated play from human play while
remaining robust to the obfuscation applied by the validator payload pipeline.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ACTION_ORDER = ["fold", "check", "call", "bet", "raise"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_ORDER)}
STREET_ORDER = ["preflop", "flop", "turn", "river", "showdown"]
STREET_INDEX = {s: i for i, s in enumerate(STREET_ORDER)}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent


def _extract_hand_action_features(actions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Behavioral features derived from a single hand's action sequence."""
    n = max(len(actions), 1)
    counts = Counter(str(a.get("action_type", "")).lower() for a in actions)

    total = sum(counts[a] for a in ACTION_ORDER)
    if total <= 0:
        total = 1

    fold_ratio = counts["fold"] / total
    check_ratio = counts["check"] / total
    call_ratio = counts["call"] / total
    bet_ratio = counts["bet"] / total
    raise_ratio = counts["raise"] / total

    amounts = [float(a.get("normalized_amount_bb", 0.0) or 0.0) for a in actions]
    pots_after = [float(a.get("pot_after", 0.0) or 0.0) for a in actions]
    pots_before = [float(a.get("pot_before", 0.0) or 0.0) for a in actions]

    amount_mean = float(np.mean(amounts)) if amounts else 0.0
    amount_std = float(np.std(amounts)) if amounts else 0.0
    amount_max = float(np.max(amounts)) if amounts else 0.0

    pot_growths = [
        _safe_div(pb - pa, pa, 0.0)
        for pa, pb in zip(pots_before, pots_after)
        if pa > 0
    ]
    pot_growth_mean = float(np.mean(pot_growths)) if pot_growths else 0.0
    pot_growth_std = float(np.std(pot_growths)) if pot_growths else 0.0

    # Transition counts (e.g. check -> fold, call -> raise)
    transitions: Counter = Counter()
    for prev, cur in zip(actions, actions[1:]):
        p = str(prev.get("action_type", "")).lower()
        c = str(cur.get("action_type", "")).lower()
        transitions[(p, c)] += 1

    # Regularity: how often the same action type repeats consecutively
    repeat_count = sum(1 for prev, cur in zip(actions, actions[1:]) if prev.get("action_type") == cur.get("action_type"))
    repeat_ratio = repeat_count / max(n - 1, 1)

    # Street progression entropy
    street_counts = Counter(str(a.get("street", "preflop")).lower() for a in actions)
    street_entropy = _entropy([street_counts.get(s, 0) for s in STREET_ORDER])

    # Actor diversity (bots often play mechanically from every seat)
    actor_seats = [int(a.get("actor_seat", 0) or 0) for a in actions]
    unique_actors = len(set(actor_seats))

    return {
        "hand_action_count": n,
        "hand_fold_ratio": fold_ratio,
        "hand_check_ratio": check_ratio,
        "hand_call_ratio": call_ratio,
        "hand_bet_ratio": bet_ratio,
        "hand_raise_ratio": raise_ratio,
        "hand_amount_mean": amount_mean,
        "hand_amount_std": amount_std,
        "hand_amount_max": amount_max,
        "hand_pot_growth_mean": pot_growth_mean,
        "hand_pot_growth_std": pot_growth_std,
        "hand_repeat_ratio": repeat_ratio,
        "hand_street_entropy": street_entropy,
        "hand_unique_actors": unique_actors,
        "hand_aggression_ratio": _safe_div(raise_ratio + bet_ratio, call_ratio + check_ratio, 0.0),
        "hand_passivity_ratio": _safe_div(call_ratio + check_ratio, raise_ratio + bet_ratio + fold_ratio, 0.0),
    }


def _extract_hand_meta_features(hand: Dict[str, Any]) -> Dict[str, float]:
    """Static/table features for a single hand."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    max_seats = int(metadata.get("max_seats", 0) or 0)
    player_count = len(players)
    street_count = len(streets)
    showdown = 1.0 if outcome.get("showdown") else 0.0

    stacks = [float(p.get("starting_stack", 0.0) or 0.0) for p in players]
    stack_mean = float(np.mean(stacks)) if stacks else 0.0
    stack_std = float(np.std(stacks)) if stacks else 0.0

    return {
        "hand_max_seats": max_seats,
        "hand_player_count": player_count,
        "hand_street_count": street_count,
        "hand_showdown": showdown,
        "hand_stack_mean": stack_mean,
        "hand_stack_std": stack_std,
        "hand_seat_fill_ratio": _safe_div(player_count, max_seats, 0.0),
    }


def _aggregate(values: List[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        arr = np.asarray([0.0])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return a fixed-length feature dictionary for a chunk of hands."""
    if not chunk:
        return {name: 0.0 for name in FEATURE_NAMES}

    action_feats_list: List[Dict[str, float]] = []
    meta_feats_list: List[Dict[str, float]] = []

    for hand in chunk:
        if not isinstance(hand, dict):
            hand = {}
        action_feats_list.append(_extract_hand_action_features(hand.get("actions") or []))
        meta_feats_list.append(_extract_hand_meta_features(hand))

    # Aggregate per-hand action features across the chunk
    aggregated: Dict[str, float] = {}
    for key in action_feats_list[0].keys():
        values = [f[key] for f in action_feats_list]
        aggregated.update(_aggregate(values, f"chunk_{key}"))

    # Aggregate per-hand meta features across the chunk
    for key in meta_feats_list[0].keys():
        values = [f[key] for f in meta_feats_list]
        aggregated.update(_aggregate(values, f"chunk_{key}"))

    # Cross-hand consistency / regularity features
    action_count_values = [f["hand_action_count"] for f in action_feats_list]
    raise_ratio_values = [f["hand_raise_ratio"] for f in action_feats_list]
    fold_ratio_values = [f["hand_fold_ratio"] for f in action_feats_list]
    amount_std_values = [f["hand_amount_std"] for f in action_feats_list]

    aggregated["chunk_action_count_cv"] = _safe_div(
        float(np.std(action_count_values)),
        float(np.mean(action_count_values)),
        0.0,
    )
    aggregated["chunk_raise_ratio_cv"] = _safe_div(
        float(np.std(raise_ratio_values)),
        float(np.mean(raise_ratio_values)),
        0.0,
    )
    aggregated["chunk_fold_ratio_cv"] = _safe_div(
        float(np.std(fold_ratio_values)),
        float(np.mean(fold_ratio_values)),
        0.0,
    )
    aggregated["chunk_amount_std_mean"] = float(np.mean(amount_std_values))

    # Global action distribution over the whole chunk
    all_action_counts: Counter = Counter()
    for hand in chunk:
        for action in (hand.get("actions") or []):
            all_action_counts[str(action.get("action_type", "")).lower()] += 1

    total_actions = sum(all_action_counts[a] for a in ACTION_ORDER) or 1
    for action in ACTION_ORDER:
        aggregated[f"chunk_global_{action}_ratio"] = all_action_counts[action] / total_actions

    aggregated["chunk_global_action_entropy"] = _entropy(
        [all_action_counts[a] for a in ACTION_ORDER]
    )
    aggregated["chunk_hand_count"] = float(len(chunk))

    return aggregated


def feature_vector(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """Return a NumPy feature vector for ``chunk`` in a deterministic order."""
    feats = extract_chunk_features(chunk)
    return np.asarray([feats[name] for name in FEATURE_NAMES], dtype=float)


def feature_names() -> Tuple[str, ...]:
    return tuple(FEATURE_NAMES)


# Build the canonical feature-name list once so training and inference agree.
_DUMMY_HAND: Dict[str, Any] = {
    "metadata": {"game_type": "holdem", "limit_type": "nl", "max_seats": 6, "hero_seat": 1},
    "players": [{"seat": 1, "starting_stack": 100.0}],
    "streets": [{"street": "preflop"}],
    "actions": [{"action_type": "call", "actor_seat": 1, "normalized_amount_bb": 1.0, "pot_before": 1.0, "pot_after": 2.0}],
    "outcome": {"showdown": True},
}
_dummy = extract_chunk_features([_DUMMY_HAND])
FEATURE_NAMES: List[str] = sorted(_dummy.keys())
