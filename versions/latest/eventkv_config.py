"""Configuration for the feature/event/persistent-KV streaming policy.

This module deliberately has no torch/model imports.  Its region capacities
are separate: short memory is bounded by ``event_window`` frames, long feature
memory by ``event_budget``, and archived visual KV by ``kv_budget`` per layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping, Optional


@dataclass
class EventKVConfig:
    recent_budget: int = 50
    recent_fallback_budget: int = 400
    event_budget: int = 200
    text_budget: int = 2000
    event_text_max_tokens: int = 100
    kv_budget: int = 1000
    # Keep the original overflow policy available for controlled baselines.
    # Merge depth starts at zero and is independent of constituent-event count.
    retire_policy: str = "merge_depth"
    retire_merge_depth: int = 5
    retire_similarity_threshold: float = 0.8
    time_cost_mode: str = "original"
    event_window: int = 8
    min_event_frames: int = 2
    short_merge_rounds: int = 2
    similarity_weight: float = 0.4
    time_weight: float = 0.2
    merge_weight: float = 0.4
    summary_cost_threshold: float = 0.5
    query_window: int = 32
    v2v_layer: Optional[int] = None
    summary_max_new_tokens: int = 64
    # Event-local summaries use only the protected system prefix and the
    # explicitly supplied event/notes. Historical context remains available
    # as an explicit controlled baseline.
    summary_context_scope: str = "event_only"
    novelty_reference: str = "recent_mean"
    novelty_reference_chunks: int = 4
    w_n: float = 0.3
    w_d: float = 0.4
    w_a: float = 0.3

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]] = None) -> "EventKVConfig":
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("memory.eventkv must be a mapping")
        known = {item.name for item in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown memory.eventkv fields: {sorted(unknown)}")
        result = cls(**dict(value))
        result.validate()
        return result

    def validate(self) -> None:
        positive_ints = (
            "recent_budget", "recent_fallback_budget", "event_budget", "kv_budget",
            "event_window", "min_event_frames", "short_merge_rounds", "query_window",
            "summary_max_new_tokens", "event_text_max_tokens", "novelty_reference_chunks",
            "retire_merge_depth",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"memory.eventkv.{name} must be a positive integer")
        if not isinstance(self.retire_policy, str) or self.retire_policy not in {"merge_depth", "overflow", "depth_or_similarity"}:
            raise ValueError("memory.eventkv.retire_policy must be merge_depth, overflow or depth_or_similarity")
        if not isinstance(self.time_cost_mode, str) or self.time_cost_mode not in {"original", "streamforest"}:
            raise ValueError("memory.eventkv.time_cost_mode must be original or streamforest")
        if not isinstance(self.summary_context_scope, str) or self.summary_context_scope not in {"history", "event_only"}:
            raise ValueError("memory.eventkv.summary_context_scope must be history or event_only")
        value = self.retire_similarity_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("memory.eventkv.retire_similarity_threshold must be finite in [-1, 1]")
        if isinstance(self.text_budget, bool) or not isinstance(self.text_budget, int) or self.text_budget < 0:
            raise ValueError("memory.eventkv.text_budget must be a nonnegative integer")
        if self.event_text_max_tokens < self.summary_max_new_tokens:
            raise ValueError("memory.eventkv.event_text_max_tokens must be >= summary_max_new_tokens")
        if self.event_window < 3 or not 2 <= self.min_event_frames < self.event_window:
            raise ValueError("memory.eventkv requires 2 <= min_event_frames < event_window")
        if self.recent_fallback_budget < self.recent_budget:
            raise ValueError("memory.eventkv.recent_fallback_budget must be >= recent_budget")
        if self.v2v_layer is not None and (
            isinstance(self.v2v_layer, bool) or not isinstance(self.v2v_layer, int) or self.v2v_layer < 0
        ):
            raise ValueError("memory.eventkv.v2v_layer must be a nonnegative integer or null")
        if self.novelty_reference not in {"previous", "recent_mean"}:
            raise ValueError("memory.eventkv.novelty_reference must be previous or recent_mean")
        for name in ("similarity_weight", "time_weight", "merge_weight", "w_n", "w_d", "w_a"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"memory.eventkv.{name} must be finite and nonnegative")
        for names in (("similarity_weight", "time_weight", "merge_weight"), ("w_n", "w_d", "w_a")):
            if not math.isclose(sum(getattr(self, name) for name in names), 1.0, abs_tol=1e-6):
                raise ValueError(f"memory.eventkv weights {names} must sum to one")
        value = self.summary_cost_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("memory.eventkv.summary_cost_threshold must be finite in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
