"""Device-resident visual tiers for the EventKV streaming memory.

Matching and merge priorities follow the official StreamForest implementation:
https://github.com/MCG-NJU/StreamForest/blob/main/llava/model/multimodal_projector/memory_manager.py

Changes for the agreed EventKV flow: an eight-frame short queue without the
additional eighteen-frame reserve; two per-frame half merges; persistent merge
weights; native Qwen representative positions; and alternating long-event
merge / oldest-event retirement.  This module never copies visual payloads to
CPU and never keeps an evicted frame or retired event as a hidden backup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class VisualFrame:
    features: torch.Tensor
    positions: torch.Tensor
    novelty: torch.Tensor
    frame_id: int
    time: float
    weights: Optional[torch.Tensor] = None
    # Similarity across the ORIGINAL frame boundary, before token merging.
    similarity_from_previous: Optional[float] = None

    @property
    def token_count(self) -> int:
        return int(self.features.shape[0])


@dataclass
class VisualEvent:
    event_id: int
    features: torch.Tensor
    positions: torch.Tensor
    novelty: torch.Tensor
    weights: torch.Tensor
    start_time: float
    end_time: float
    merge_count: int = 1
    summary_ids: list[int] = field(default_factory=list)
    frame_count: int = 0
    representative_time: Optional[float] = None

    @property
    def token_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def midpoint(self) -> float:
        if self.representative_time is not None:
            return float(self.representative_time)
        return (float(self.start_time) + float(self.end_time)) / 2.0


def _validate_payload(features, positions, novelty, weights=None) -> None:
    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("visual features must be a nonempty [tokens, hidden] tensor")
    count = features.shape[0]
    if positions.shape != (3, count):
        raise ValueError("visual positions must have shape [3, tokens]")
    if novelty.shape != (count,):
        raise ValueError("visual novelty must have shape [tokens]")
    tensors = [positions, novelty] + ([] if weights is None else [weights])
    if any(t.device != features.device for t in tensors):
        raise ValueError("visual payload and metadata must share a device")
    if weights is not None and weights.shape != (count,):
        raise ValueError("visual weights must have shape [tokens]")
    if weights is not None and bool((weights <= 0).any()):
        raise ValueError("visual weights must be positive")


@torch.no_grad()
def merge_half(
    features: torch.Tensor,
    positions: torch.Tensor,
    novelty: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One StreamForest/ToMe bipartite half merge, retaining Qwen positions.

    The descriptor is the full projected visual vector.  StreamForest's
    SigLIP-specific head reshape has no equivalent semantic meaning for Qwen
    projected embeddings.  Destination coordinates survive unchanged; source
    mass and N are averaged with persistent weights, with accumulation in FP32.
    The output is stably sorted by representative temporal coordinate.
    """
    _validate_payload(features, positions, novelty, weights)
    count = int(features.shape[0])
    if weights is None:
        weights = torch.ones(count, dtype=torch.float32, device=features.device)
    else:
        weights = weights.float()
    if count == 1:
        # Compact storage is important when the input is a slice of a much
        # larger retired tensor. Detach alone would retain that storage.
        return (features.detach().clone(), positions.detach().clone(),
                novelty.detach().float().clone(), weights.detach().clone())

    metric = F.normalize(features.float(), dim=-1, eps=1e-8)
    source_ids = torch.arange(0, count, 2, device=features.device)
    destination_ids = torch.arange(1, count, 2, device=features.device)
    similarities = metric[source_ids] @ metric[destination_ids].T
    best_scores, best_destinations = similarities.max(dim=-1)
    ranking = torch.argsort(best_scores, descending=True, stable=True)
    remove_count = count // 2
    merged_sources = ranking[:remove_count]
    unmerged_sources = ranking[remove_count:]
    destinations = best_destinations[merged_sources]
    source_rows = source_ids[merged_sources]
    unmerged_rows = source_ids[unmerged_sources]

    destination_weights = weights[destination_ids].clone()
    destination_weights.index_add_(0, destinations, weights[source_rows])
    values = features[destination_ids].float() * weights[destination_ids, None]
    values.index_add_(0, destinations, features[source_rows].float() * weights[source_rows, None])
    values /= destination_weights[:, None]
    n_values = novelty[destination_ids].float() * weights[destination_ids]
    n_values.index_add_(0, destinations, novelty[source_rows].float() * weights[source_rows])
    n_values /= destination_weights

    new_features = torch.cat((features[unmerged_rows], values.to(features.dtype)), dim=0)
    new_positions = torch.cat((positions[:, unmerged_rows], positions[:, destination_ids]), dim=1)
    new_novelty = torch.cat((novelty[unmerged_rows].float(), n_values), dim=0)
    new_weights = torch.cat((weights[unmerged_rows], destination_weights), dim=0)
    order = torch.argsort(new_positions[0], stable=True)
    return (new_features[order].contiguous(), new_positions[:, order].contiguous(),
            new_novelty[order].contiguous(), new_weights[order].contiguous())


@torch.no_grad()
def frame_similarity(previous: VisualFrame, current: VisualFrame) -> float:
    """StreamForest's original-frame flatten cosine, with variable-grid fallback.

    Qwen allows different patch counts per frame.  A mean descriptor is used
    only when flattening would produce unequal vector lengths.
    """
    if previous.features.shape == current.features.shape:
        left, right = previous.features.flatten(), current.features.flatten()
    else:
        left, right = previous.features.float().mean(0), current.features.float().mean(0)
    return float(F.cosine_similarity(left.float(), right.float(), dim=0, eps=1e-8).item())


@torch.no_grad()
def event_match_similarity(left: VisualEvent, right: VisualEvent) -> float:
    """StreamForest's best bipartite-edge mean, rather than centroid cosine.

    As in the reference, matching is run on the joint token list and can also
    select pairs within a constituent event. It measures mergeability.
    """
    values = torch.cat((left.features, right.features), dim=0)
    metric = F.normalize(values.float(), dim=-1, eps=1e-8)
    scores = metric[::2] @ metric[1::2].T
    best = scores.max(dim=-1).values
    return float(best.topk(values.shape[0] // 2).values.mean().item())


class EventVectorMemory:
    """Whole-frame Recent -> bounded short queue -> long-event visual memory.

    ``on_retire`` consumes an event synchronously to create persistent KV; the
    memory does not retain that event afterwards. ``on_summary`` sees the two
    BEFORE-merge events and returns only new compensation text to append to
    their existing notes. ``on_rewrite`` receives the complete concatenated
    notes and both source events when a single event's text exceeds its limit.
    """

    def __init__(self, config, on_retire: Optional[Callable[[VisualEvent], Any]] = None,
                 on_summary: Optional[Callable[[VisualEvent, VisualEvent, float], list[int]]] = None,
                 on_rewrite: Optional[Callable[[list[int], VisualEvent, VisualEvent], list[int]]] = None):
        self.config = config
        self.on_retire = on_retire
        self.on_summary = on_summary
        self.on_rewrite = on_rewrite
        if int(config.event_window) < 3:
            raise ValueError("event_window must contain at least three frames")
        if not 1 <= int(config.min_event_frames) < int(config.event_window):
            raise ValueError("min_event_frames must be below event_window")
        self.reset()

    def reset(self) -> None:
        self.recent: list[VisualFrame] = []
        self.short: list[VisualFrame] = []
        self.long_events: list[VisualEvent] = []
        self.effective_recent_budget = int(self.config.recent_budget)
        self.current_time = 0.0
        self._last_frame_id: Optional[int] = None
        self._next_event_id = 0
        self.operation_log = deque(maxlen=128)
        self.counters = dict(frames_seen=0, short_compressions=0, formed_events=0,
                             merged_events=0, retired_events=0, summary_requests=0,
                             text_rewrite_requests=0,
                             recent_fallbacks=0, peak_recent_tokens=0,
                             peak_short_frames=0, peak_short_tokens=0,
                             peak_event_tokens=0, peak_resident_bytes=0)

    @property
    def recent_tokens(self) -> int:
        return sum(f.token_count for f in self.recent)

    @property
    def short_tokens(self) -> int:
        return sum(f.token_count for f in self.short)

    @property
    def event_tokens(self) -> int:
        return sum(e.token_count for e in self.long_events)

    @staticmethod
    def _payload_bytes(value: VisualFrame | VisualEvent) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (value.features, value.positions, value.novelty, value.weights)
                   if t is not None)

    def resident_bytes(self) -> int:
        return sum(self._payload_bytes(v) for v in self.recent + self.short + self.long_events)

    def _observe_peaks(self) -> None:
        for name, value in (("peak_recent_tokens", self.recent_tokens),
                            ("peak_short_frames", len(self.short)),
                            ("peak_short_tokens", self.short_tokens),
                            ("peak_event_tokens", self.event_tokens),
                            ("peak_resident_bytes", self.resident_bytes())):
            self.counters[name] = max(self.counters[name], value)

    @torch.no_grad()
    def add_frame(self, frame: VisualFrame) -> None:
        _validate_payload(frame.features, frame.positions, frame.novelty, frame.weights)
        if self._last_frame_id is not None and int(frame.frame_id) <= self._last_frame_id:
            raise ValueError("frames must arrive in strictly increasing frame_id order")
        if not math.isfinite(float(frame.time)) or (self.counters["frames_seen"] and frame.time < self.current_time):
            raise ValueError("frame time must be finite and nondecreasing")
        if frame.token_count > self.effective_recent_budget:
            fallback = int(self.config.recent_fallback_budget)
            if frame.token_count > fallback:
                raise ValueError(f"complete frame needs {frame.token_count} Recent tokens, exceeding fallback {fallback}")
            self.effective_recent_budget = fallback
            self.counters["recent_fallbacks"] += 1
        # HF can return per-image slices of one batched allocation. Own a
        # compact frame allocation, so retaining the newest frame cannot keep
        # already-retired raw frames alive through a shared backing storage.
        frame = VisualFrame(frame.features.detach().clone(), frame.positions.detach().clone(),
                            frame.novelty.detach().float().clone(), int(frame.frame_id),
                            float(frame.time), None if frame.weights is None else
                            frame.weights.detach().float().clone(),
                            frame.similarity_from_previous)
        if self.recent:
            frame.similarity_from_previous = frame_similarity(self.recent[-1], frame)
        self.recent.append(frame)
        self.current_time = float(frame.time)
        self._last_frame_id = int(frame.frame_id)
        self.counters["frames_seen"] += 1
        self._observe_peaks()

        while self.recent_tokens > self.effective_recent_budget:
            old = self.recent.pop(0)
            features, positions, novelty, weights = old.features, old.positions, old.novelty, old.weights
            frame_id, frame_time = old.frame_id, old.time
            boundary_similarity = old.similarity_from_previous
            # Do not pin the retired full-resolution frame across subsequent
            # event/LLM callbacks after its compressed replacement exists.
            del old
            for _ in range(int(self.config.short_merge_rounds)):
                features, positions, novelty, weights = merge_half(features, positions, novelty, weights)
            self.short.append(VisualFrame(features, positions, novelty, frame_id, frame_time,
                                          weights, boundary_similarity))
            self.counters["short_compressions"] += 1
            self._observe_peaks()
            self._split_ready_events()
        self._observe_peaks()

    def _split_ready_events(self) -> None:
        window = int(self.config.event_window)
        while len(self.short) >= window:
            first_cut = int(self.config.min_event_frames)
            # Eight FRAMES contain seven inner boundaries. The reference uses
            # eight EDGES; here follow the explicitly requested frame window.
            candidates = range(first_cut, window)
            cut = min(candidates, key=lambda index: (
                self.short[index].similarity_from_previous
                if self.short[index].similarity_from_previous is not None else 1.0,
                index,
            ))
            self._form_event(cut)

    def _form_event(self, frame_count: int) -> None:
        frames = self.short[:frame_count]
        if not frames:
            return
        features = torch.cat([f.features for f in frames], dim=0)
        positions = torch.cat([f.positions for f in frames], dim=1)
        novelty = torch.cat([f.novelty for f in frames], dim=0)
        weights = torch.cat([f.weights if f.weights is not None else
                             torch.ones(f.token_count, device=f.features.device) for f in frames], dim=0)
        features, positions, novelty, weights = merge_half(features, positions, novelty, weights)
        event = VisualEvent(self._next_event_id, features, positions, novelty, weights,
                            frames[0].time, frames[-1].time, frame_count=len(frames))
        self._next_event_id += 1
        del self.short[:frame_count]
        self.long_events.append(event)
        self.counters["formed_events"] += 1
        self.operation_log.append(dict(kind="form", event_id=event.event_id,
                                       frames=len(frames), tokens=event.token_count))
        # The short rows are absent from self.short now. Drop their local
        # owners before a retirement callback can start a sizeable prefill.
        del frames
        self._observe_peaks()
        self._enforce_event_budget()

    @torch.no_grad()
    def flush_short(self) -> None:
        """Seal the remaining short tail at question time, including one frame."""
        self._split_ready_events()
        if self.short:
            self._form_event(len(self.short))

    def _merge_costs(self) -> list[float]:
        similarities = [event_match_similarity(a, b)
                        for a, b in zip(self.long_events, self.long_events[1:])]
        merged_counts = [(a.merge_count + b.merge_count) / 2.0
                         for a, b in zip(self.long_events, self.long_events[1:])]
        max_count = max(merged_counts)
        # This is the reference's mean representative age position, NOT a
        # timestamp difference despite the original variable name time_diffs.
        normalizer = max(float(self.current_time), 1e-8)
        return [float(self.config.similarity_weight) * (1.0 - similarity)
                + float(self.config.time_weight) * ((left.midpoint + right.midpoint) / 2.0) / normalizer
                + float(self.config.merge_weight) * (count / max_count + 1e-6)
                for left, right, similarity, count in
                zip(self.long_events, self.long_events[1:], similarities, merged_counts)]

    def _merge_once(self) -> None:
        costs = self._merge_costs()
        index = min(range(len(costs)), key=lambda i: (costs[i], i))
        left, right = self.long_events[index:index + 2]
        summary_ids = list(left.summary_ids) + list(right.summary_ids)
        text_tokens_before = len(summary_ids)
        text_tokens_generated = 0
        high_cost = costs[index] > float(self.config.summary_cost_threshold)
        if high_cost:
            self.counters["summary_requests"] += 1
            if self.on_summary is not None:
                new_summary_ids = list(self.on_summary(left, right, costs[index]))
                text_tokens_generated = len(new_summary_ids)
                summary_ids.extend(new_summary_ids)
        text_limit = int(getattr(self.config, "event_text_max_tokens", 100))
        text_rewritten = len(summary_ids) > text_limit
        if text_rewritten:
            self.counters["text_rewrite_requests"] += 1
            if self.on_rewrite is None:
                raise RuntimeError("event text exceeds event_text_max_tokens but on_rewrite is not configured")
            summary_ids = list(self.on_rewrite(summary_ids, left, right))
            rewrite_limit = min(text_limit, int(getattr(self.config, "summary_max_new_tokens", 64)))
            if len(summary_ids) > rewrite_limit:
                raise ValueError(f"on_rewrite returned {len(summary_ids)} tokens, exceeding limit {rewrite_limit}")
        features, positions, novelty, weights = merge_half(
            torch.cat((left.features, right.features), dim=0),
            torch.cat((left.positions, right.positions), dim=1),
            torch.cat((left.novelty, right.novelty), dim=0),
            torch.cat((left.weights, right.weights), dim=0))
        left_count = max(left.frame_count, 1)
        right_count = max(right.frame_count, 1)
        midpoint = ((left.midpoint * left_count + right.midpoint * right_count)
                    / (left_count + right_count))
        event = VisualEvent(left.event_id, features, positions, novelty, weights,
                            left.start_time, right.end_time,
                            left.merge_count + right.merge_count, summary_ids,
                            left.frame_count + right.frame_count, midpoint)
        self.long_events[index:index + 2] = [event]
        self.counters["merged_events"] += 1
        self.operation_log.append(dict(kind="merge", event_id=event.event_id,
                                       right_event_id=right.event_id,
                                       tokens_before=left.token_count + right.token_count,
                                       tokens=event.token_count, cost=costs[index],
                                       summary_requested=high_cost,
                                       text_tokens_before=text_tokens_before,
                                       text_tokens_generated=text_tokens_generated,
                                       text_tokens_after=len(summary_ids),
                                       text_rewritten=text_rewritten))

    def _retire_oldest(self) -> None:
        event = self.long_events[0]
        if self.on_retire is None:
            raise RuntimeError("event budget requires on_retire to consume persistent KV")
        self.on_retire(event)
        del self.long_events[0]
        self.counters["retired_events"] += 1
        self.operation_log.append(dict(kind="retire", event_id=event.event_id,
                                       tokens=event.token_count))

    def _enforce_event_budget(self) -> None:
        budget = int(self.config.event_budget)
        while self.event_tokens > budget:
            if len(self.long_events) >= 2:
                self._merge_once()
            if self.event_tokens > budget:
                self._retire_oldest()

    def get_stats(self) -> dict[str, Any]:
        return {**self.counters, "recent_tokens": self.recent_tokens,
                "recent_frames": len(self.recent), "short_tokens": self.short_tokens,
                "short_frames": len(self.short), "event_tokens": self.event_tokens,
                "event_count": len(self.long_events),
                "effective_recent_budget": self.effective_recent_budget,
                "vector_resident_bytes": self.resident_bytes(),
                "short_frame_limit_after_update": int(self.config.event_window) - 1}
