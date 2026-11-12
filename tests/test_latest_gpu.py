"""CPU tests of the EventKV device controller with a real tiny Qwen decoder.

No checkpoint, visual tower, CUDA allocation, or benchmark is needed.  The
tests check native prefill equivalence, frozen event rankings across repeated
budget reductions, and restoration/release of a question's temporary suffix.
"""
from __future__ import annotations
import copy
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import types
import pytest
import torch
from versions.latest.eventkv_config import EventKVConfig
from versions.latest.eventkv_memory import VisualEvent, VisualFrame

def _tiny_text_model():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    torch.manual_seed(419)
    config = Qwen2_5_VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=32, rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}, layer_types=['full_attention'] * 2, attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    return Qwen2_5_VLTextModel(config).eval()

class _Tokenizer:
    eos_token_id = 31
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        ids = [1 + ord(character) % 29 for character in str(text)] or [1]
        if kwargs.get('return_tensors') == 'pt':
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return ' '.join((str(int(token)) for token in ids))

class _TinyQA:

    def __init__(self, model):
        self.language_model = model
        self.config = model.config
        self.num_layers = len(model.layers)
        self.visual_start_idx = 3
        self.device = torch.device('cpu')
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())
        self.lm_head = torch.nn.Linear(32, 32, bias=False)
        self._layer_position_ids = {}
        self.conv_history = []
        with torch.inference_mode():
            self.kv_cache = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        prefix_positions = torch.arange(3).float().expand(3, -1)
        self._position_ids_cache = [prefix_positions.clone() for _ in model.layers]

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def parameters(self):
        return self.language_model.parameters()

    def _ensure_dynamic_cache(self):
        from transformers import DynamicCache
        if self.kv_cache is not None and (not isinstance(self.kv_cache, DynamicCache)):
            self.kv_cache = DynamicCache.from_legacy_cache(self.kv_cache)

    def _get_cache_seq_len_per_layer(self):
        return [int(k.shape[-2]) for k, _ in self.kv_cache]

    def _get_next_global_offset_per_layer(self):
        return [int(positions.max()) + 1 if positions is not None and positions.numel() else 0 for positions in self._position_ids_cache]

    def _build_position_ids_3d_for_vision(self, positions, batch):
        return positions.unsqueeze(1).expand(3, batch, -1).clone()

    def _build_position_ids_3d_for_text(self, offset, count, batch):
        return torch.arange(offset, offset + count).float().view(1, 1, -1).expand(3, batch, -1).clone()

    def _append_position_ids_layer_explicit(self, layer, positions):
        old = self._position_ids_cache[layer]
        self._position_ids_cache[layer] = positions.clone() if old is None else torch.cat((old, positions), 1)

def _controller(**overrides):
    from versions.latest.eventkv_gpu import EventKVGPUIntegrator
    values = dict(recent_budget=64, recent_fallback_budget=64, event_budget=200, text_budget=0, kv_budget=1000, query_window=2)
    values.update(overrides)
    config = EventKVConfig(**values)
    model = _tiny_text_model()
    qa = _TinyQA(model)
    wrapper = SimpleNamespace(qa_model=qa, qa_processor=qa.processor, sample_fps=0.5, profiler=SimpleNamespace(phase=lambda *args, **kwargs: nullcontext()))
    method_config = SimpleNamespace(method='eventkv_stream', eventkv=config)
    return (EventKVGPUIntegrator(wrapper, method_config), qa, config)

def _positions(count, offset):
    positions = torch.arange(offset, offset + count).float()
    return torch.stack((torch.full_like(positions, float(offset)), positions, positions.remainder(4)))

def _event(event_id, count=16):
    generator = torch.Generator().manual_seed(123 + event_id)
    features = torch.randn(count, 32, generator=generator)
    return VisualEvent(event_id=event_id, features=features, positions=_positions(count, 10 + 30 * event_id), novelty=torch.linspace(0, 1, count), weights=torch.ones(count), start_time=2.0 * event_id, end_time=2.0 * event_id + 1)

def _cache_copy(qa):
    return [(k.clone(), v.clone()) for k, v in qa.kv_cache]

def _assert_compact(tensor):
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()

def test_embedding_prefill_matches_unmodified_qwen_and_keeps_original_positions():
    controller, qa, _ = _controller()
    native = copy.deepcopy(qa.language_model)
    for layer in native.layers:
        layer.self_attn.config = native.config
    features = _event(1, 7).features
    positions = _positions(7, 41)
    native_cache = copy.deepcopy(qa.kv_cache)
    try:
        with torch.inference_mode():
            expected = native(inputs_embeds=features.unsqueeze(0), position_ids=positions.unsqueeze(1), past_key_values=native_cache, use_cache=True)
            controller._append_embeddings(features, positions)
        for (actual_k, actual_v), (full_k, full_v) in zip(qa.kv_cache, expected.past_key_values):
            torch.testing.assert_close(actual_k, full_k, atol=0, rtol=0)
            torch.testing.assert_close(actual_v, full_v, atol=0, rtol=0)
        for stored in qa._position_ids_cache:
            torch.testing.assert_close(stored[:, -7:], positions)
    finally:
        controller.close()

def test_depth_retirement_prefills_real_kv_and_uses_unchanged_s_compression():
    controller, qa, config = _controller(event_budget=10, kv_budget=4)
    try:
        assert config.retire_policy == 'merge_depth' and config.retire_merge_depth == 5
        left, right = (_event(0, 8), _event(1, 8))
        left.merge_count, left.merge_depth = (5, 4)
        controller.memory.current_time = 10.0
        controller.memory.long_events = [left, right]
        controller.memory._merge_costs = lambda: [0.0]
        controller.memory._enforce_event_budget()
        assert not controller.memory.long_events
        stats = controller.get_stats()
        assert stats['retired_by_merge_depth'] == 1
        assert stats['effective_kv_compressions'] == 1
        assert stats['persistent_visual_kv_tokens_by_layer'] == [4, 4]
        assert all((event.score_builds == 1 for event in controller.kv_events))
        assert qa._get_cache_seq_len_per_layer() == [7, 7]
    finally:
        controller.close()

def test_immature_depth_policy_keeps_overflow_management_out_of_persistent_kv():
    controller, qa, _ = _controller(event_budget=10, kv_budget=4)
    try:
        controller.memory.long_events = [_event(i, 8) for i in range(3)]
        controller.memory._merge_costs = lambda: [0.0] * (len(controller.memory.long_events) - 1)
        controller.memory._enforce_event_budget()
        stats = controller.get_stats()
        assert stats['event_tokens'] == 8
        assert stats['long_event_merge_depths'] == [2]
        assert stats['retired_events'] == 0
        assert stats['persistent_visual_kv_tokens_by_layer'] == [0, 0]
        assert stats['effective_kv_compressions'] == 0
        assert qa._get_cache_seq_len_per_layer() == [3, 3]
    finally:
        controller.close()

def test_cold_pretrained_loader_initializes_prefix_before_controller_attachment():
    from versions.latest.eventkv_gpu import EventKVGPUIntegrator
    model = _tiny_text_model()
    qa = _TinyQA(model)
    qa.kv_cache = None
    calls = []

    def initialize_prefix():
        calls.append(True)
        with torch.inference_mode():
            qa.kv_cache = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
    qa.encode_init_prompt = initialize_prefix
    wrapper = SimpleNamespace(qa_model=qa, qa_processor=qa.processor, sample_fps=0.5)
    config = SimpleNamespace(eventkv=EventKVConfig())
    controller = EventKVGPUIntegrator(wrapper, config)
    try:
        assert calls == [True]
        assert controller.prefix_len == 3
        assert controller._cache_lengths() == [3, 3]
        controller.retire_event(_event(0, 4))
        assert controller._cache_lengths() == [7, 7]
    finally:
        controller.close()

def test_repeated_kv_budget_reductions_build_each_event_score_once():
    controller, qa, config = _controller()
    try:
        controller.retire_event(_event(0))
        controller.retire_event(_event(1))
        assert qa._get_cache_seq_len_per_layer() == [35, 35]
        config.kv_budget = 4
        controller.compact_kv()
        assert qa._get_cache_seq_len_per_layer() == [7, 7]
        assert controller.kv_events
        assert all((event.score_builds == 1 for event in controller.kv_events))
        builds = {event.event_id: event.score_builds for event in controller.kv_events}
        config.kv_budget = 2
        controller.compact_kv()
        assert qa._get_cache_seq_len_per_layer() == [5, 5]
        assert all((event.score_builds == builds[event.event_id] == 1 for event in controller.kv_events))
        for k, v in qa.kv_cache:
            _assert_compact(k)
            _assert_compact(v)
        for positions, (k, _) in zip(qa._position_ids_cache, qa.kv_cache):
            assert positions.shape[-1] == k.shape[-2]
            _assert_compact(positions)
    finally:
        controller.close()

def test_question_suffix_is_removed_and_old_kv_storage_is_really_released():
    controller, qa, _ = _controller()
    try:
        controller.retire_event(_event(0, 8))
        frame = VisualFrame(features=_event(2, 12).features, positions=_positions(12, 90), novelty=torch.full((12,), 0.5), frame_id=0, time=10.0)
        controller.memory.add_frame(frame)
        expected_cache = _cache_copy(qa)
        expected_positions = [positions.clone() for positions in qa._position_ids_cache]
        for _ in range(2):
            controller.prepare_answer()
            assert min(qa._get_cache_seq_len_per_layer()) > 11
            offset = max(qa._get_next_global_offset_per_layer())
            controller._append_embeddings(torch.randn(3, 32), _positions(3, offset))
            controller.finish_answer()
            for actual, expected in zip(qa.kv_cache, expected_cache):
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
                    _assert_compact(actual_tensor)
            for actual, expected in zip(qa._position_ids_cache, expected_positions):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                _assert_compact(actual)
    finally:
        controller.close()

def test_summary_queries_are_text_queries_without_persistent_text_kv():
    controller, qa, config = _controller(text_budget=100)
    try:
        event = _event(0, 8)
        event.summary_ids = [4, 5, 6]
        controller.retire_event(event)
        state = controller.kv_events[0]
        assert state.query_source == 'text'
        assert state.query.shape == (1, 4, 3, 8)
        assert state.query_positions.shape == (3,)
        assert qa._get_cache_seq_len_per_layer() == [11, 11]
        assert state.visual_query.shape == (1, 4, 2, 8)
        assert all((values.shape == (8,) for values in state.d_by_layer))
        config.text_budget = 1
        controller._enforce_text_budget()
        assert state.summary_ids == [6]
        assert state.query.shape[-2] == 1
        config.text_budget = 0
        controller._enforce_text_budget()
        assert state.query_source == 'visual'
        assert state.query.shape[-2] == 2
        torch.testing.assert_close(state.query, state.visual_query)
        config.kv_budget = 4
        controller.compact_kv()
        assert state.score_builds == 1
        assert state.query is None and state.visual_query is None
        assert controller.get_stats()['query_cache_bytes'] == 0
        assert qa._get_cache_seq_len_per_layer() == [7, 7]
    finally:
        controller.close()

def test_layerwise_rankings_keep_different_ids_in_one_absolute_position_frame():
    controller, qa, config = _controller()
    try:
        controller.retire_event(_event(0, 8))
        state = controller.kv_events[0]
        state.novelty = torch.zeros_like(state.novelty)
        state.query = torch.zeros_like(state.query)
        state.d_by_layer = [torch.linspace(0, 1, 8), torch.linspace(1, 0, 8)]
        original = _cache_copy(qa)
        original_positions = [positions.clone() for positions in qa._position_ids_cache]
        controller._build_score(state)
        expected_scores = [{int(uid): float(score) for uid, score in zip(ids, scores)} for ids, scores in zip(state.token_ids_by_layer, state.scores_by_layer)]
        config.kv_budget = 4
        controller.compact_kv()
        assert not torch.equal(state.token_ids_by_layer[0], state.token_ids_by_layer[1])
        assert qa._get_cache_seq_len_per_layer() == [7, 7]
        for layer, ids in enumerate(state.token_ids_by_layer):
            indices = torch.cat((torch.arange(3), 3 + ids))
            for actual, full in zip(qa.kv_cache[layer], original[layer]):
                torch.testing.assert_close(actual, full.index_select(-2, indices), atol=0, rtol=0)
            torch.testing.assert_close(qa._position_ids_cache[layer], original_positions[layer].index_select(-1, indices), atol=0, rtol=0)
            for uid, score in zip(ids, state.scores_by_layer[layer]):
                assert float(score) == expected_scores[layer][int(uid)]
        controller.retire_event(_event(1, 3))
        assert qa._get_cache_seq_len_per_layer() == [7, 7]
        assert state.score_builds == 1
        assert all((event.score_builds == 1 for event in controller.kv_events))
        assert all((torch.isfinite(tensor).all() for pair in qa.kv_cache for tensor in pair))
        for layer, ids in enumerate(state.token_ids_by_layer):
            for uid, score in zip(ids, state.scores_by_layer[layer]):
                assert float(score) == expected_scores[layer][int(uid)]
    finally:
        controller.close()

def test_partial_answer_prefill_failure_restores_the_persistent_prefix(monkeypatch):
    controller, qa, _ = _controller()
    try:
        controller.retire_event(_event(0, 8))
        controller.memory.long_events.append(_event(1, 6))
        controller.memory.add_frame(VisualFrame(features=_event(2, 12).features, positions=_positions(12, 90), novelty=torch.full((12,), 0.5), frame_id=0, time=10.0))
        expected_cache = _cache_copy(qa)
        original = controller._append_embeddings
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('simulated prefill failure')
            return original(*args, **kwargs)
        monkeypatch.setattr(controller, '_append_embeddings', interrupted)
        with pytest.raises(RuntimeError, match='simulated prefill failure'):
            controller.prepare_answer()
        assert controller._answer_lengths is None
        for actual, expected in zip(qa.kv_cache, expected_cache):
            for actual_tensor, expected_tensor in zip(actual, expected):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
                _assert_compact(actual_tensor)
    finally:
        controller.close()

def test_an_earlier_event_query_cannot_score_later_event_keys():
    controller, qa, _ = _controller()
    try:
        controller.retire_event(_event(0, 8))
        controller.retire_event(_event(1, 8))
        state = controller.kv_events[0]
        identical_probe = copy.deepcopy(state)
        controller._build_score(state)
        with torch.inference_mode():
            anchor_keys = qa.kv_cache[controller.anchor_layer][0]
            anchor_keys[:, :, 3 + 8:].fill_(100000)
        controller._build_score(identical_probe)
        for expected, actual in zip(state.scores_by_layer, identical_probe.scores_by_layer):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        controller.close()

def test_real_tiny_visual_tower_streams_pil_frames_through_all_memory_tiers():
    """Exercise actual HF preprocessing, visual encoding, and decoder prefill."""
    from PIL import Image
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from versions.latest.eventkv_gpu import EventKVGPUIntegrator
    text_config = _tiny_text_model().config.to_dict()
    config = Qwen2_5_VLConfig(text_config=text_config, vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=4, patch_size=2, spatial_merge_size=2, temporal_patch_size=2, window_size=8, out_hidden_size=32, fullatt_block_indexes=[0, 1]))
    config._attn_implementation = 'sdpa'
    torch.manual_seed(509)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    qa = _TinyQA(model.language_model)
    qa.config = model.config
    qa.get_image_features = model.get_image_features

    class Processor:
        tokenizer = _Tokenizer()

        def __init__(self):
            self.image_processor = Qwen2VLImageProcessor(min_pixels=16, max_pixels=256, patch_size=2, temporal_patch_size=2, merge_size=2)

        def __call__(self, *, images, text, return_tensors):
            return self.image_processor(images=images, return_tensors=return_tensors)

    class FixedTokenHead(torch.nn.Module):

        def forward(self, hidden):
            logits = hidden.new_zeros((*hidden.shape[:-1], 32))
            logits[..., 7] = 1
            return logits
    qa.processor = Processor()
    qa.lm_head = FixedTokenHead()
    wrapper = SimpleNamespace(qa_model=qa, qa_processor=qa.processor, sample_fps=0.5)
    policy = EventKVConfig(recent_budget=16, recent_fallback_budget=16, event_budget=2, text_budget=16, kv_budget=4, query_window=2, summary_cost_threshold=0, summary_max_new_tokens=1, retire_policy='overflow')
    controller = EventKVGPUIntegrator(wrapper, SimpleNamespace(method='eventkv_stream', eventkv=policy))
    try:
        frames = [Image.new('RGB', (8, 8), (0, 0, 0)) for _ in range(2)]
        frames += [Image.new('RGB', (16, 16), (255, 255, 255)) for _ in range(14)]
        for start in range(0, len(frames), 2):
            result = controller.observe_chunk(frames[start:start + 2], seconds=4.0)
            assert result['encoding'] == 'image'
            assert len(controller.memory.short) < 8
            assert controller.memory.event_tokens <= 2
            assert max((len(ids) for ids in controller.live_ids_by_layer)) <= 4
        stats = controller.get_stats()
        assert stats['formed_events'] > 0
        assert stats['merged_events'] > 0
        assert stats['retired_events'] > 0
        assert stats['summary_generation_calls'] > 0
        assert stats['summary_generated_tokens'] > 0
        assert stats['effective_kv_compressions'] > 0
        assert stats['text_tokens'] <= 16
        assert all((event.score_builds <= 1 for event in controller.kv_events))
        controller.prepare_answer()
        old_lengths = list(controller._answer_lengths)
        expected = [(k[:, :, :length].clone(), v[:, :, :length].clone()) for (k, v), length in zip(qa.kv_cache, old_lengths)]
        try:
            hidden = controller._append_embeddings(controller._embed_text([4, 5]), controller._text_positions(2))
            assert int(qa.lm_head(hidden[:, -1:]).argmax(-1)) == 7
        finally:
            controller.finish_answer()
        assert qa._get_cache_seq_len_per_layer() == old_lengths
        for actual, full in zip(qa.kv_cache, expected):
            for tensor, before in zip(actual, full):
                torch.testing.assert_close(tensor, before, atol=0, rtol=0)
                _assert_compact(tensor)
    finally:
        controller.close()

def test_stats_remain_available_after_the_runtime_clears_the_model_cache():
    controller, qa, _ = _controller()
    try:
        controller.retire_event(_event(0, 8))
        assert controller.get_stats()['kv_bytes'] > 0
        qa.kv_cache = None
        stats = controller.get_stats()
        assert stats['kv_bytes'] == 0
        assert stats['peak_kv_bytes'] > 0
        assert stats['memory_method'] == 'eventkv_stream'
    finally:
        controller.close()
