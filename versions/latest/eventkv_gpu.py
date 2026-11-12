"""Event vectors and persistent KV; native attention with read-only Q capture."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Optional

import torch
import torch.nn.functional as F

from .eventkv_memory import EventVectorMemory, VisualFrame, VisualEvent
from eventkv.signals import incoming_attention_scores, layer_evolution_scores


def _normalize(x):
    x = x.float()
    if not x.numel():
        return x
    span = x.max() - x.min()
    return torch.where(span > 1e-8, (x - x.min()) / span.clamp_min(1e-8), torch.zeros_like(x))


@dataclass
class KVEventState:
    event_id: int
    token_ids_by_layer: list[torch.Tensor]
    novelty: Optional[torch.Tensor]
    d_by_layer: Optional[list[torch.Tensor]]
    query: Optional[torch.Tensor]
    query_positions: Optional[torch.Tensor]
    start_time: float
    end_time: float
    summary_ids: list[int]
    scores_by_layer: Optional[list[torch.Tensor]] = None
    score_builds: int = 0
    visual_query: Optional[torch.Tensor] = None
    visual_query_positions: Optional[torch.Tensor] = None
    query_source: str = "visual"


@dataclass
class TextMemory:
    event_id: int
    start_time: float
    end_time: float
    summary_ids: list[int]


def _storage_bytes(tensors):
    """Count physical storage once, including aliased Q/fallback views."""
    storages = {}
    for tensor in tensors:
        if tensor is not None and tensor.numel():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class EventKVGPUIntegrator:
    method_name = "eventkv_stream"

    def __init__(self, wrapper, method_config):
        from eventkv.attention_capture import QueryCaptureAdapter

        self.wrapper = wrapper
        self.qa_model = wrapper.qa_model
        self.config = method_config.eventkv
        self.config.validate()
        # The pretrained loader defers prefix prefill until a stream starts.
        # Attachment happens earlier, so initialize the same official prompt
        # before reading cache device/lengths or installing capture hooks.
        if self.qa_model.kv_cache is None:
            self.qa_model.encode_init_prompt()
        self.layers = self.qa_model.language_model.layers
        self.num_layers = len(self.layers)
        self.anchor_layer = (self.num_layers // 2 if self.config.v2v_layer is None
                             else int(self.config.v2v_layer))
        if not 0 <= self.anchor_layer < self.num_layers:
            raise ValueError("eventkv.v2v_layer is outside the decoder")
        self._handles = []
        self._capture_d = False
        self._capture_queries = False
        self._previous_hidden = None
        self._last_query = None
        self._d_values = []
        self._install_d_hooks()
        self._attention_adapter = QueryCaptureAdapter(self.layers, query_capture=self._capture_q)
        self._original_prune = getattr(self.qa_model, "apply_kv_cache_pruning_strict", None)
        self._original_offsets = self.qa_model._get_next_global_offset_per_layer
        self.qa_model.apply_kv_cache_pruning_strict = self.gather_absolute
        self.qa_model._get_next_global_offset_per_layer = self.next_global_offsets
        self.memory = EventVectorMemory(
            self.config, self.retire_event, self._make_summary, self._rewrite_summary)
        self.reset()

    def _install_d_hooks(self):
        def begin(module, args, kwargs):
            if self._capture_d:
                self._previous_hidden = (args[0] if args else kwargs["hidden_states"]).detach()

        self._handles.append(self.layers[0].register_forward_pre_hook(begin, with_kwargs=True))
        for index, layer in enumerate(self.layers):
            def capture(module, args, output, layer_index=index):
                if self._capture_d:
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._d_values.append(layer_evolution_scores(
                        self._previous_hidden, hidden).reshape(-1).detach())
                    self._previous_hidden = hidden.detach()
            self._handles.append(layer.register_forward_hook(capture))

    def _capture_q(self, layer_index, query):
        if self._capture_queries and layer_index == self.anchor_layer:
            self._last_query = query.detach().clone()

    def reset(self):
        self.memory.reset()
        self.prefix_len = int(self.qa_model.visual_start_idx)
        sample = self.qa_model.kv_cache[0][0]
        self.device = sample.device
        self.live_ids_by_layer = [torch.empty(0, device=self.device, dtype=torch.long)
                                  for _ in self.layers]
        self.kv_events: list[KVEventState] = []
        self.orphan_summaries: list[TextMemory] = []
        self._chunk_history = deque(maxlen=self.config.novelty_reference_chunks)
        self._next_token_id = 0
        self._next_frame_id = 0
        self._stream_time = 0.0
        self._next_visual_position = max(self.next_global_offsets())
        self._answer_lengths = None
        self._capture_d = self._capture_queries = False
        self._previous_hidden = self._last_query = None
        self._d_values = []
        self.events = []
        self.counters = dict(kv_compression_triggers=0, effective_kv_compressions=0,
                             removed_kv_tokens_per_layer=0, scored_events=0,
                             summary_generation_calls=0, summary_generation_seconds=0.0,
                             summary_generated_tokens=0, stream_encoding_seconds=0.0,
                             summary_append_calls=0, summary_append_seconds=0.0,
                             summary_append_generated_tokens=0,
                             summary_rewrite_calls=0, summary_rewrite_seconds=0.0,
                             summary_rewrite_generated_tokens=0,
                             event_prefill_seconds=0.0, answer_prefill_seconds=0.0,
                             peak_persistent_visual_kv_tokens=0, peak_kv_overshoot=0,
                             peak_kv_bytes=0, first_effective_compression_seconds=None,
                             effective_compression_seconds=0.0)

    def close(self):
        self._attention_adapter.close()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_prune is not None:
            self.qa_model.apply_kv_cache_pruning_strict = self._original_prune
        else:
            del self.qa_model.apply_kv_cache_pruning_strict
        self.qa_model._get_next_global_offset_per_layer = self._original_offsets

    def next_global_offsets(self):
        # All layers use the same absolute position frame, even when different
        # token identities survived in each layer. Never rebase retained keys.
        values = getattr(self.qa_model, "_position_ids_cache", [])
        offset = max((int(p.max().item()) + 1 for p in values
                      if p is not None and p.numel()), default=0)
        return [offset] * self.num_layers

    def _timestamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.inference_mode()
    def _append_embeddings(self, features, positions, capture=False):
        if not features.shape[0]:
            return
        self.qa_model._ensure_dynamic_cache()
        if positions.shape != (3, features.shape[0]):
            raise ValueError("embedding/position length mismatch")
        previous = self._capture_d, self._capture_queries
        self._capture_d = bool(capture)
        self._capture_queries = bool(capture) or previous[1]
        if capture:
            self._d_values = []
            self._last_query = None
        try:
            out = self.qa_model.language_model(
                inputs_embeds=features.unsqueeze(0),
                position_ids=positions.unsqueeze(1),
                past_key_values=self.qa_model.kv_cache, use_cache=True, return_dict=True,
            )
            self.qa_model.kv_cache = out.past_key_values
            for layer in range(self.num_layers):
                self.qa_model._append_position_ids_layer_explicit(layer, positions)
            self._observe_cache_peak()
            return out.last_hidden_state
        finally:
            self._capture_d, self._capture_queries = previous
            self._previous_hidden = None

    def _text_positions(self, count):
        # Local summary caches exclude old visual keys but keep the same
        # absolute text positions as the historical-context baseline.
        offset = max(max(self.next_global_offsets()),
                     getattr(self, "_summary_position_floor", 0))
        return torch.arange(offset, offset + count, device=self.device,
                            dtype=torch.float32).expand(3, -1).clone()

    def _embed_text(self, ids):
        token_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        return self.qa_model.get_input_embeddings()(token_ids)[0]

    @torch.inference_mode()
    def retire_event(self, event: VisualEvent):
        started = self._timestamp()
        count = event.token_count
        ids = torch.arange(self._next_token_id, self._next_token_id + count,
                           device=self.device, dtype=torch.long)
        self._next_token_id += count
        self._append_embeddings(event.features, event.positions, capture=True)
        d_values = self._d_values
        self._d_values = []
        if len(d_values) != self.num_layers or self._last_query is None:
            raise RuntimeError("event prefill failed to capture N/D/Q scoring inputs")
        width = min(count, self.config.query_window)
        visual_query = self._last_query[:, :, -width:].clone()
        visual_query_positions = (ids[-width:] * 2).clone()
        query, query_positions = visual_query, visual_query_positions
        summary_ids = list(event.summary_ids)[-self.config.text_budget:] if self.config.text_budget else []
        source = "visual"
        if summary_ids:
            lengths = self._cache_lengths()
            try:
                self._capture_queries = True
                self._append_embeddings(self._embed_text(summary_ids), self._text_positions(len(summary_ids)))
                query = self._last_query.detach().clone()
                # Causal order tags are separate from the M-RoPE coordinates.
                # Every text query follows this event and precedes later ones.
                query_positions = torch.full((len(summary_ids),), int(ids[-1].item()) * 2 + 1,
                                             device=self.device, dtype=torch.long)
                source = "text"
            finally:
                self._capture_queries = False
                self._restore_lengths(lengths)
        self._last_query = None
        self.kv_events.append(KVEventState(
            event.event_id, [ids.clone() for _ in self.layers],
            event.novelty.detach().float().clone(), d_values, query, query_positions,
            event.start_time, event.end_time, summary_ids,
            visual_query=visual_query, visual_query_positions=visual_query_positions,
            query_source=source,
        ))
        for layer in range(self.num_layers):
            self.live_ids_by_layer[layer] = torch.cat((self.live_ids_by_layer[layer], ids))
        self.counters["event_prefill_seconds"] += self._timestamp() - started
        self.compact_kv()

    def _build_score(self, event):
        if event.scores_by_layer is not None:
            return
        anchor_ids = event.token_ids_by_layer[self.anchor_layer]
        live_ids = self.live_ids_by_layer[self.anchor_layer]
        keys = self.qa_model.kv_cache[self.anchor_layer][0]
        prefix_positions = torch.arange(-self.prefix_len, 0, device=self.device) * 2
        key_order = torch.cat((prefix_positions, live_ids * 2))
        attention, _ = incoming_attention_scores(
            event.query, keys, event.query_positions, key_order, causal=True,
        )
        physical = self.prefix_len + torch.searchsorted(live_ids, anchor_ids)
        a = _normalize(attention[physical])
        n = _normalize(event.novelty)
        event.scores_by_layer = [
            (self.config.w_n * n + self.config.w_d * _normalize(d)
             + self.config.w_a * a).detach().clone() for d in event.d_by_layer
        ]
        event.score_builds += 1
        self.counters["scored_events"] += 1
        event.novelty = event.d_by_layer = None
        event.query = event.query_positions = None
        event.visual_query = event.visual_query_positions = None

    @torch.inference_mode()
    def compact_kv(self):
        if max((ids.numel() for ids in self.live_ids_by_layer), default=0) <= self.config.kv_budget:
            return
        self.counters["kv_compression_triggers"] += 1
        started = self._timestamp()
        # One live-state scoring snapshot precedes deletions. Historical fixed
        # rankings are intentionally NOT recalculated in subsequent rounds.
        for event in self.kv_events:
            self._build_score(event)
        round_started = started
        while max(ids.numel() for ids in self.live_ids_by_layer) > self.config.kv_budget:
            before = max(ids.numel() for ids in self.live_ids_by_layer)
            for event in self.kv_events:
                for layer in range(self.num_layers):
                    scores = event.scores_by_layer[layer]
                    count = scores.numel()
                    keep = (count + 1) // 2 if count > 1 else 0
                    selected = torch.argsort(scores, descending=True, stable=True)[:keep].sort().values
                    event.token_ids_by_layer[layer] = event.token_ids_by_layer[layer][selected].clone()
                    event.scores_by_layer[layer] = scores[selected].clone()
            survivors = [torch.cat([e.token_ids_by_layer[layer] for e in self.kv_events])
                         for layer in range(self.num_layers)]
            indices = [torch.cat((torch.arange(self.prefix_len, device=self.device),
                                  self.prefix_len + torch.searchsorted(old, new)))
                       for old, new in zip(self.live_ids_by_layer, survivors)]
            self.qa_model.apply_kv_cache_pruning_strict(indices)
            self.live_ids_by_layer = survivors
            for event in self.kv_events:
                if not event.token_ids_by_layer[0].numel() and event.summary_ids:
                    # Text has its own sliding lifetime; deleting the last KV
                    # of an event need not erase its already generated notes.
                    self.orphan_summaries.append(TextMemory(
                        event.event_id, event.start_time, event.end_time, event.summary_ids))
            self.kv_events = [e for e in self.kv_events if e.token_ids_by_layer[0].numel()]
            after = max(ids.numel() for ids in survivors)
            self.counters["effective_kv_compressions"] += 1
            self.counters["removed_kv_tokens_per_layer"] += before - after
            self.events.append(dict(kind="kv_half", visual_before=before, visual_after=after,
                                    evicted=before - after, score_recomputed=False))
            round_ended = self._timestamp()
            self.events[-1]["duration_seconds"] = round_ended - round_started
            if self.counters["first_effective_compression_seconds"] is None:
                self.counters["first_effective_compression_seconds"] = round_ended - round_started
            round_started = round_ended
        elapsed = round_started - started
        self.counters["effective_compression_seconds"] += elapsed

    def _cache_lengths(self):
        if self.qa_model.kv_cache is None:
            return [0] * self.num_layers
        return [int(k.shape[-2]) for k, _ in self.qa_model.kv_cache]

    @torch.inference_mode()
    def gather_absolute(self, keep_indices_by_layer):
        from transformers import DynamicCache
        cache, positions = [], []
        for layer, ((key, value), keep) in enumerate(zip(self.qa_model.kv_cache, keep_indices_by_layer)):
            keep = torch.as_tensor(keep, device=key.device, dtype=torch.long)
            if keep.numel() < self.prefix_len or not torch.equal(
                    keep[:self.prefix_len], torch.arange(self.prefix_len, device=key.device)):
                raise ValueError("EventKV pruning must retain the protected prefix")
            cache.append((key.index_select(-2, keep), value.index_select(-2, keep)))
            positions.append(self.qa_model._position_ids_cache[layer].index_select(-1, keep))
        self.qa_model.kv_cache = DynamicCache.from_legacy_cache(tuple(cache))
        self.qa_model._position_ids_cache = positions

    def _restore_lengths(self, lengths):
        # Clone rather than slice: a surviving view would pin all discarded KV.
        from transformers import DynamicCache
        compact = [(k[:, :, :length].clone(), v[:, :, :length].clone())
                   for (k, v), length in zip(self.qa_model.kv_cache, lengths)]
        self.qa_model.kv_cache = DynamicCache.from_legacy_cache(tuple(compact))
        self.qa_model._position_ids_cache = [p[:, :length].clone()
                                             for p, length in zip(self.qa_model._position_ids_cache, lengths)]
        self.qa_model._layer_position_ids.clear()

    @torch.inference_mode()
    def observe_chunk(self, frames, *, seconds=None):
        started = self._timestamp()
        if not frames:
            return {"frames": 0, "cache_after": max(self._cache_lengths())}
        qa = self.qa_model
        dtype = next(qa.language_model.parameters()).dtype
        encoded = qa.processor(text=[""], images=[f.convert("RGB") for f in frames],
                               return_tensors="pt").to(self.device, dtype)
        grids = encoded["image_grid_thw"]
        parts = qa.get_image_features(encoded["pixel_values"], grids)
        self.wrapper.last_encode_style = "image"
        merge = int(qa.config.vision_config.spatial_merge_size)
        counts = [int(grid.prod().item()) // (merge * merge) for grid in grids]
        if isinstance(parts, torch.Tensor):
            parts = list(parts.reshape(-1, parts.shape[-1]).split(counts))
        else:
            parts = list(parts)
        if len(parts) != len(frames):
            raise RuntimeError("EventKV requires one independent image feature group per frame")
        pooled = sum((p.float().sum(0) for p in parts)) / sum(counts)
        if self._chunk_history:
            reference = (self._chunk_history[-1] if self.config.novelty_reference == "previous"
                         else torch.stack(list(self._chunk_history)).mean(0))
            novelty = (1 - F.cosine_similarity(pooled, reference, dim=0)).clamp(0, 1)
        else:
            novelty = pooled.new_zeros(())
        self._chunk_history.append(pooled.detach().clone())
        times = getattr(frames, "frame_times", None)
        fps = float(self.wrapper.sample_fps)
        for index, (part, grid) in enumerate(zip(parts, grids)):
            t, h, w = [int(x) for x in grid.tolist()]
            h, w = h // merge, w // merge
            row = torch.arange(h, device=self.device).view(1, h, 1).expand(t, h, w).flatten()
            col = torch.arange(w, device=self.device).view(1, 1, w).expand(t, h, w).flatten()
            positions = torch.stack((torch.zeros_like(row), row, col)).float() + self._next_visual_position
            self._next_visual_position = int(positions.max().item()) + 1
            frame_time = float(times[index]) if times is not None else self._stream_time + index / fps
            self.memory.add_frame(VisualFrame(
                part.detach(), positions, novelty.expand(part.shape[0]).clone(),
                self._next_frame_id, frame_time,
            ))
            self._next_frame_id += 1
            self._enforce_text_budget()
        self._stream_time += float(seconds if seconds is not None else len(frames) / fps)
        self.counters["stream_encoding_seconds"] += self._timestamp() - started
        return {"frames": len(frames), "cache_after": max(self._cache_lengths()),
                "encoding": "image", "memory_method": self.method_name}

    def _enforce_text_budget(self):
        entries = sorted(list(self.kv_events) + list(self.memory.long_events) + self.orphan_summaries,
                         key=lambda entry: (entry.end_time, entry.start_time))
        excess = sum(len(e.summary_ids) for e in entries) - self.config.text_budget
        for entry in entries:
            if excess <= 0:
                break
            remove = min(excess, len(entry.summary_ids))
            if not remove:
                continue
            entry.summary_ids = entry.summary_ids[remove:]
            excess -= remove
            if isinstance(entry, KVEventState) and entry.query_source == "text" and entry.query is not None:
                if entry.summary_ids:
                    entry.query = entry.query[:, :, remove:].clone()
                    entry.query_positions = entry.query_positions[remove:].clone()
                else:
                    entry.query = entry.visual_query
                    entry.query_positions = entry.visual_query_positions
                    entry.query_source = "visual"
        self.orphan_summaries = [entry for entry in self.orphan_summaries if entry.summary_ids]

    @torch.inference_mode()
    def _make_summary(self, left, right, cost):
        """Generate extra loss-compensation notes; the caller keeps old notes."""
        if self.config.text_budget == 0:
            return []
        tokenizer = self.qa_model.processor.tokenizer
        prior = "\n".join(tokenizer.decode(event.summary_ids, skip_special_tokens=True)
                          for event in (left, right) if event.summary_ids)
        limit = self._memory_text_generation_limit()
        prompt = (
            f"\nWrite additional factual notes about the observed visual events from "
            f"{left.start_time:.1f}s to {right.end_time:.1f}s. "
            "Preserve observed actions, entities, order, and changes missing from the existing notes. "
            "The existing notes will be kept separately: do not rewrite or repeat them. "
            "Do not invent details. Use short complete sentences, "
            f"at most {limit} tokens, and stop when complete.\n"
            f"Existing notes:\n{prior}\nAdditional factual event notes: ")
        return self._generate_memory_text(prompt, visual_events=(left, right), kind="append")

    @torch.inference_mode()
    def _rewrite_summary(self, summary_ids, left, right):
        """Condense an overlong concatenation without changing its visual event."""
        if self.config.text_budget == 0:
            return []
        if len(summary_ids) <= self.config.event_text_max_tokens:
            return list(summary_ids)
        notes = self.qa_model.processor.tokenizer.decode(summary_ids, skip_special_tokens=True)
        limit = self._memory_text_generation_limit()
        prompt = (
            f"\nCondense only the following chronological event notes spanning "
            f"{left.start_time:.1f}s to {right.end_time:.1f}s. "
            "Preserve their factual actions, entities, order, and changes. "
            "Do not add information or repeat details. Use short complete sentences, "
            f"at most {limit} tokens, and stop when complete.\n"
            f"Event notes:\n{notes}\nCondensed factual event notes: ")
        output = self._generate_memory_text(prompt, kind="rewrite")
        if not output:
            # Preserve the existing events on failure instead of silently
            # replacing their notes with empty text or a truncated suffix.
            raise RuntimeError("event text rewrite produced no tokens; merge was not committed")
        return output

    def _memory_text_generation_limit(self):
        return min(self.config.summary_max_new_tokens,
                   self.config.event_text_max_tokens, self.config.text_budget)

    @contextmanager
    def _memory_text_context(self):
        """Give memory writing its own prefix-only cache, or the old baseline.

        The main visual cache is never copied or appended to in event_only
        mode. Only its small protected prefix is cloned on the same device.
        Restore object references, including capture metadata, even when a
        visual prefill or a decode step fails.
        """
        if self.config.summary_context_scope == "history":
            lengths = self._cache_lengths()
            try:
                yield
            finally:
                self._restore_lengths(lengths)
            return

        from transformers import DynamicCache

        qa = self.qa_model
        main_cache = qa.kv_cache
        main_positions = qa._position_ids_cache
        main_layer_positions = qa._layer_position_ids
        capture_state = (self._capture_d, self._capture_queries,
                         self._previous_hidden, self._last_query, self._d_values)
        missing = object()
        previous_floor = getattr(self, "_summary_position_floor", missing)
        previous_saved_bytes = getattr(self, "_saved_summary_main_kv_bytes", missing)
        position_floor = max(self.next_global_offsets())
        if previous_floor is not missing:
            position_floor = max(position_floor, previous_floor)
        if len(main_cache) != self.num_layers or any(
                key.shape[-2] < self.prefix_len or value.shape[-2] < self.prefix_len
                for key, value in main_cache):
            raise RuntimeError("summary context requires the complete protected prefix")
        prefix = tuple((key[:, :, :self.prefix_len].clone(),
                        value[:, :, :self.prefix_len].clone())
                       for key, value in main_cache)
        prefix_positions = [positions[:, :self.prefix_len].clone()
                            for positions in main_positions]
        temporary_cache = DynamicCache.from_legacy_cache(prefix)
        try:
            qa.kv_cache = temporary_cache
            qa._position_ids_cache = prefix_positions
            qa._layer_position_ids = {}
            self._capture_d = self._capture_queries = False
            self._previous_hidden = self._last_query = None
            self._d_values = []
            self._summary_position_floor = position_floor
            self._saved_summary_main_kv_bytes = sum(
                key.numel() * key.element_size() + value.numel() * value.element_size()
                for key, value in main_cache)
            if previous_saved_bytes is not missing:
                self._saved_summary_main_kv_bytes += previous_saved_bytes
            yield
        finally:
            qa.kv_cache = main_cache
            qa._position_ids_cache = main_positions
            qa._layer_position_ids = main_layer_positions
            (self._capture_d, self._capture_queries, self._previous_hidden,
             self._last_query, self._d_values) = capture_state
            if previous_floor is missing:
                del self._summary_position_floor
            else:
                self._summary_position_floor = previous_floor
            if previous_saved_bytes is missing:
                del self._saved_summary_main_kv_bytes
            else:
                self._saved_summary_main_kv_bytes = previous_saved_bytes

    @torch.inference_mode()
    def _generate_memory_text(self, prompt, *, visual_events=(), kind="append"):
        """Greedy generation capped during decoding, with a transient KV suffix."""
        if kind not in {"append", "rewrite"}:
            raise ValueError(f"unknown memory text generation kind: {kind}")
        limit = self._memory_text_generation_limit()
        if limit == 0:
            return []
        started = self._timestamp()
        tokenizer = self.qa_model.processor.tokenizer
        format_prompt = getattr(self.qa_model, "get_prompt", None)
        if format_prompt is not None:
            prompt = format_prompt(prompt, mc=False)
        output = []
        with self._memory_text_context():
            for event in visual_events:
                self._append_embeddings(event.features, event.positions)
            prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
            hidden = self._append_embeddings(self._embed_text(prompt_ids), self._text_positions(len(prompt_ids)))
            eos = tokenizer.eos_token_id
            stop = {int(eos)} if eos is not None else set()
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if convert is not None:
                im_end = convert("<|im_end|>")
                if im_end is not None and im_end != getattr(tokenizer, "unk_token_id", None):
                    stop.add(int(im_end))
            for _ in range(limit):
                token = int(self.qa_model.lm_head(hidden[:, -1:]).argmax(-1).item())
                if token in stop:
                    break
                output.append(token)
                hidden = self._append_embeddings(self._embed_text([token]), self._text_positions(1))
        elapsed = self._timestamp() - started
        self.counters["summary_generation_calls"] += 1
        self.counters["summary_generated_tokens"] += len(output)
        self.counters["summary_generation_seconds"] += elapsed
        self.counters[f"summary_{kind}_calls"] += 1
        self.counters[f"summary_{kind}_generated_tokens"] += len(output)
        self.counters[f"summary_{kind}_seconds"] += elapsed
        return output

    @torch.inference_mode()
    def prepare_answer(self):
        if self._answer_lengths is not None:
            raise RuntimeError("an answer is already being prepared")
        # Flush may move older events into persistent KV, so snapshot AFTER it.
        self.memory.flush_short()
        self._enforce_text_budget()
        self._answer_lengths = self._cache_lengths()
        started = self._timestamp()
        try:
            for event in sorted(self.kv_events + self.orphan_summaries, key=lambda e: e.end_time):
                if event.summary_ids:
                    self._append_embeddings(self._embed_text(event.summary_ids), self._text_positions(len(event.summary_ids)))
            for event in self.memory.long_events:
                self._append_embeddings(event.features, event.positions)
                if event.summary_ids:
                    self._append_embeddings(self._embed_text(event.summary_ids), self._text_positions(len(event.summary_ids)))
            for frame in self.memory.recent:
                self._append_embeddings(frame.features, frame.positions)
        except BaseException:
            self.finish_answer()
            raise
        self.counters["answer_prefill_seconds"] += self._timestamp() - started

    def finish_answer(self):
        if self._answer_lengths is not None:
            lengths, self._answer_lengths = self._answer_lengths, None
            self._restore_lengths(lengths)

    def _observe_cache_peak(self):
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for k, v in self.qa_model.kv_cache)
        # The event-local writer keeps the untouched main cache resident while
        # a separate temporary cache is active; both count toward KV memory.
        size += getattr(self, "_saved_summary_main_kv_bytes", 0)
        self.counters["peak_kv_bytes"] = max(self.counters["peak_kv_bytes"], size)
        if self._answer_lengths is None and self._capture_d:
            count = max(self._cache_lengths()) - self.prefix_len
            self.counters["peak_persistent_visual_kv_tokens"] = max(
                self.counters["peak_persistent_visual_kv_tokens"], count)
            self.counters["peak_kv_overshoot"] = max(
                self.counters["peak_kv_overshoot"], count - self.config.kv_budget)

    def get_stats(self):
        cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                          for k, v in (self.qa_model.kv_cache or []))
        q_bytes = _storage_bytes(t for e in self.kv_events
                                 for t in (e.query, e.query_positions, e.visual_query, e.visual_query_positions))
        score_bytes = sum(t.numel() * t.element_size() for e in self.kv_events
                          for t in ((e.scores_by_layer or []) + (e.d_by_layer or [])
                                    + ([] if e.novelty is None else [e.novelty])))
        text_tokens = sum(len(e.summary_ids) for e in self.kv_events + self.memory.long_events + self.orphan_summaries)
        id_bytes = _storage_bytes(self.live_ids_by_layer + [ids for e in self.kv_events for ids in e.token_ids_by_layer])
        stats = {**self.memory.get_stats(), **self.counters,
                 "memory_method": self.method_name, "persistent_kv_events": len(self.kv_events),
                 "summary_context_scope": self.config.summary_context_scope,
                 "persistent_visual_kv_tokens_by_layer": [int(ids.numel()) for ids in self.live_ids_by_layer],
                 "text_tokens": text_tokens, "kv_bytes": cache_bytes,
                 "query_cache_bytes": q_bytes, "score_metadata_bytes": score_bytes,
                 "token_identity_bytes": id_bytes,
                 "novelty_history_bytes": sum(g.numel() * g.element_size() for g in self._chunk_history),
                 "kv_budget": self.config.kv_budget, "event_budget": self.config.event_budget,
                 "compression_events": self.counters["effective_kv_compressions"],
                 "compression_events_detail": list(self.events),
                 "peak_kv_tokens": self.prefix_len + self.counters["peak_persistent_visual_kv_tokens"],
                 "total_removed_tokens": self.counters["removed_kv_tokens_per_layer"],
                 "ram_scope": "unsupported_for_merged_visual_features",
                 "aore_scope": "unsupported_without_merged_feature_provenance"}
        count = self.counters["effective_kv_compressions"]
        stats["mean_effective_compression_seconds"] = (self.counters["effective_compression_seconds"] / count
                                                       if count else None)
        stats["resident_tensor_bytes"] = (stats["vector_resident_bytes"] + cache_bytes + q_bytes
                                           + score_bytes + id_bytes + stats["novelty_history_bytes"])
        return stats
