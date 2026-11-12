"""CPU semantics of event-local summary writing with a real tiny Qwen decoder.

Tests use existing tiny-Qwen fixtures, never checkpoints or CUDA. They assert
what every prefill can see, restoration of original state objects on failures,
and independence from changed persistent visual history.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
import torch
from versions.latest.eventkv_config import EventKVConfig
_spec = importlib.util.spec_from_file_location('_eventkv_context_shared', Path(__file__).with_name('test_latest_gpu.py'))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
_controller, _event, _cache_copy = (_shared._controller, _shared._event, _shared._cache_copy)

class RecordingHead(torch.nn.Module):

    def __init__(self, tokens=(7, 31), fail_at=None):
        super().__init__()
        self.tokens, self.fail_at = (tuple(tokens), fail_at)
        self.calls, self.hidden = (0, [])

    def forward(self, hidden):
        if self.calls == self.fail_at:
            raise RuntimeError('controlled decode failure')
        self.hidden.append(hidden.detach().clone())
        token = self.tokens[min(self.calls, len(self.tokens) - 1)]
        self.calls += 1
        logits = hidden.new_zeros((*hidden.shape[:-1], 32))
        logits[..., token] = 1
        return logits

def snapshot(controller, qa):
    return dict(cache=qa.kv_cache, position_list=qa._position_ids_cache, layer_positions=qa._layer_position_ids, cache_values=_cache_copy(qa), position_values=[p.clone() for p in qa._position_ids_cache], position_floor=(hasattr(controller, '_summary_position_floor'), getattr(controller, '_summary_position_floor', None)), saved_main_bytes=(hasattr(controller, '_saved_summary_main_kv_bytes'), getattr(controller, '_saved_summary_main_kv_bytes', None)), capture=(controller._capture_d, controller._capture_queries, controller._previous_hidden, controller._last_query, controller._d_values))

def assert_main_untouched(controller, qa, before):
    assert qa.kv_cache is before['cache']
    assert qa._position_ids_cache is before['position_list']
    assert qa._layer_position_ids is before['layer_positions']
    assert (hasattr(controller, '_summary_position_floor'), getattr(controller, '_summary_position_floor', None)) == before['position_floor']
    assert (hasattr(controller, '_saved_summary_main_kv_bytes'), getattr(controller, '_saved_summary_main_kv_bytes', None)) == before['saved_main_bytes']
    for actual, expected in zip(qa.kv_cache, before['cache_values']):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)
    for a, e in zip(qa._position_ids_cache, before['position_values']):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    actual_capture = (controller._capture_d, controller._capture_queries, controller._previous_hidden, controller._last_query, controller._d_values)
    for actual, expected in zip(actual_capture, before['capture']):
        assert actual is expected

def seed_capture_state(controller, qa):
    qa._layer_position_ids = {'main_only': torch.tensor([19.0])}
    controller._capture_d = controller._capture_queries = True
    controller._previous_hidden = torch.tensor([23.0])
    controller._last_query = torch.tensor([29.0])
    controller._d_values = [torch.tensor([31.0])]

@pytest.mark.parametrize('scope', ['', 'local', None, True, 0, [], {}])
def test_invalid_summary_context_is_rejected(scope):
    with pytest.raises(ValueError, match='memory.eventkv.summary_context_scope'):
        EventKVConfig.from_dict({'summary_context_scope': scope})

@pytest.mark.parametrize('tokens,expected', [((31,), []), ((7, 8, 31), [7, 8]), ((7,), [7, 7, 7])])
def test_event_only_isolates_every_append_and_restores_main_objects(tokens, expected, monkeypatch):
    controller, qa, _ = _controller(text_budget=2000, summary_max_new_tokens=3, summary_context_scope='event_only')
    try:
        controller.retire_event(_event(0, 6))
        seed_capture_state(controller, qa)
        before = snapshot(controller, qa)
        prefix = controller.prefix_len
        visual = (_event(1, 4), _event(2, 5))
        visual_before = [(e.features.clone(), e.positions.clone()) for e in visual]
        original_append = controller._append_embeddings
        seen = []

        def append(features, positions, capture=False):
            assert qa.kv_cache is not before['cache']
            assert qa._position_ids_cache is not before['position_list']
            assert qa._layer_position_ids is not before['layer_positions']
            assert not controller._capture_d and (not controller._capture_queries)
            lengths = controller._cache_lengths()
            seen.append((lengths, features.shape[0]))
            if len(seen) == 1:
                assert lengths == [prefix] * controller.num_layers
                for current, prior in zip(qa.kv_cache, before['cache_values']):
                    for c, p in zip(current, prior):
                        torch.testing.assert_close(c, p[:, :, :prefix], atol=0, rtol=0)
                        assert c.untyped_storage().nbytes() == c.numel() * c.element_size()
            else:
                previous_lengths, count = seen[-2]
                assert lengths == [length + count for length in previous_lengths]
            if len(seen) <= 2:
                torch.testing.assert_close(positions, visual[len(seen) - 1].positions, atol=0, rtol=0)
            return original_append(features, positions, capture=capture)

        def forbidden_restore(_):
            raise AssertionError('event-only writing must restore references, not copy historical KV')
        monkeypatch.setattr(controller, '_append_embeddings', append)
        monkeypatch.setattr(controller, '_restore_lengths', forbidden_restore)
        qa.lm_head = RecordingHead(tokens)
        assert controller._generate_memory_text('Additional notes:', visual_events=visual) == expected
        assert len(seen) == 3 + len(expected)
        assert_main_untouched(controller, qa, before)
        for e, (features, positions) in zip(visual, visual_before):
            torch.testing.assert_close(e.features, features, atol=0, rtol=0)
            torch.testing.assert_close(e.positions, positions, atol=0, rtol=0)
    finally:
        controller.close()

@pytest.mark.parametrize('fail_stage', ['left', 'right', 'prompt', 'decode'])
def test_event_only_restores_state_when_any_generation_phase_fails(fail_stage, monkeypatch):
    controller, qa, _ = _controller(text_budget=2000, summary_context_scope='event_only')
    try:
        controller.retire_event(_event(0, 6))
        seed_capture_state(controller, qa)
        before = snapshot(controller, qa)
        calls = []
        original_append = controller._append_embeddings
        fail_index = {'left': 0, 'right': 1, 'prompt': 2}.get(fail_stage)

        def append(*args, **kwargs):
            assert qa.kv_cache is not before['cache']
            index = len(calls)
            calls.append(index)
            if index == fail_index:
                raise RuntimeError('controlled prefill failure')
            return original_append(*args, **kwargs)
        monkeypatch.setattr(controller, '_append_embeddings', append)
        qa.lm_head = RecordingHead((7,), fail_at=1 if fail_stage == 'decode' else None)
        with pytest.raises(RuntimeError, match='controlled .* failure'):
            controller._generate_memory_text('Event notes:', visual_events=(_event(1, 4), _event(2, 4)))
        assert_main_untouched(controller, qa, before)
        assert controller.counters['summary_generation_calls'] == 0
    finally:
        controller.close()

def test_overlong_rewrite_also_starts_with_prefix_only_and_keeps_no_event_visuals(monkeypatch):
    controller, qa, _ = _controller(text_budget=2000, summary_context_scope='event_only')
    try:
        controller.retire_event(_event(0, 6))
        before = snapshot(controller, qa)
        seen = []
        original_append = controller._append_embeddings

        def append(features, positions, **kwargs):
            if not seen:
                assert controller._cache_lengths() == [controller.prefix_len] * controller.num_layers
            seen.append(features.shape[0])
            return original_append(features, positions, **kwargs)
        monkeypatch.setattr(controller, '_append_embeddings', append)
        qa.lm_head = RecordingHead((7, 31))
        assert controller._rewrite_summary([4] * 101, _event(1, 4), _event(2, 4)) == [7]
        assert len(seen) == 2 and seen[0] > 101 and (seen[1] == 1)
        assert_main_untouched(controller, qa, before)
        assert controller.counters['summary_rewrite_calls'] == 1
    finally:
        controller.close()

@pytest.mark.parametrize('scope,should_match', [('event_only', True), ('history', False)])
def test_local_summary_hidden_is_independent_of_changed_persistent_history(scope, should_match):
    collected = []
    for older_event_id in (0, 13):
        controller, qa, _ = _controller(text_budget=2000, summary_context_scope=scope)
        try:
            historical = _event(0, 6)
            historical.features = _event(older_event_id, 6).features
            controller.retire_event(historical)
            qa.lm_head = head = RecordingHead((31,))
            controller._generate_memory_text('Additional notes:', visual_events=(_event(20, 4), _event(21, 4)))
            collected.append(head.hidden[0])
        finally:
            controller.close()
    if should_match:
        torch.testing.assert_close(collected[0], collected[1], atol=0, rtol=0)
    else:
        assert not torch.allclose(collected[0], collected[1], atol=1e-07, rtol=1e-07)

def test_history_baseline_retains_historical_context_and_original_restore_behavior(monkeypatch):
    controller, qa, _ = _controller(text_budget=2000, summary_context_scope='history')
    try:
        controller.retire_event(_event(0, 6))
        before = snapshot(controller, qa)
        lengths = controller._cache_lengths()
        original_append, original_restore = (controller._append_embeddings, controller._restore_lengths)
        seen, restored = ([], [])

        def append(*args, **kwargs):
            if not seen:
                assert qa.kv_cache is before['cache']
                assert controller._cache_lengths() == lengths
            seen.append(True)
            return original_append(*args, **kwargs)

        def restore(value):
            restored.append(value)
            return original_restore(value)
        monkeypatch.setattr(controller, '_append_embeddings', append)
        monkeypatch.setattr(controller, '_restore_lengths', restore)
        qa.lm_head = RecordingHead((7, 31))
        assert controller._generate_memory_text('Event notes:', visual_events=(_event(1, 4),)) == [7]
        assert restored == [lengths]
        assert qa.kv_cache is not before['cache']
        for actual, expected in zip(qa.kv_cache, before['cache_values']):
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
        assert controller.counters['summary_append_generated_tokens'] == 1
    finally:
        controller.close()

@pytest.mark.parametrize('kind', ['append', 'rewrite'])
def test_isolation_preserves_all_absolute_visual_and_text_positions_from_history(kind, monkeypatch):
    observed = []
    for scope in ('history', 'event_only'):
        controller, qa, _ = _controller(text_budget=2000, summary_context_scope=scope)
        try:
            controller.retire_event(_event(13, 6))
            original_append = controller._append_embeddings
            positions_seen = []

            def append(features, positions, **kwargs):
                positions_seen.append(positions.clone())
                return original_append(features, positions, **kwargs)
            monkeypatch.setattr(controller, '_append_embeddings', append)
            qa.lm_head = RecordingHead((7, 8, 31))
            visual = (_event(1, 4), _event(2, 4)) if kind == 'append' else ()
            controller._generate_memory_text('Event notes:', visual_events=visual, kind=kind)
            observed.append(positions_seen)
            assert not hasattr(controller, '_summary_position_floor')
        finally:
            controller.close()
    assert len(observed[0]) == len(observed[1])
    for history, local in zip(*observed):
        torch.testing.assert_close(local, history, atol=0, rtol=0)

@pytest.mark.parametrize('fail', [False, True])
def test_nested_position_floor_value_is_restored_after_success_or_failure(fail):
    controller, qa, _ = _controller(text_budget=2000, summary_context_scope='event_only')
    try:
        controller.retire_event(_event(0, 6))
        controller._summary_position_floor = 777
        before = snapshot(controller, qa)
        qa.lm_head = RecordingHead((7, 31), fail_at=0 if fail else None)
        if fail:
            with pytest.raises(RuntimeError, match='controlled decode failure'):
                controller._generate_memory_text('Event notes:')
        else:
            assert controller._generate_memory_text('Event notes:') == [7]
        assert_main_untouched(controller, qa, before)
        assert controller._summary_position_floor == 777
    finally:
        controller.close()

@pytest.mark.parametrize('fail', [False, True])
def test_local_peak_counts_simultaneously_resident_main_and_temporary_cache(fail, monkeypatch):
    controller, qa, _ = _controller(text_budget=2000, summary_context_scope='event_only')
    try:
        controller.retire_event(_event(0, 6))
        main_bytes = sum((k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in qa.kv_cache))
        before = snapshot(controller, qa)
        observed = []
        original_observe = controller._observe_cache_peak

        def observe():
            temporary_bytes = sum((k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in qa.kv_cache))
            assert controller._saved_summary_main_kv_bytes == main_bytes
            assert qa.kv_cache is not before['cache']
            original_observe()
            observed.append(main_bytes + temporary_bytes)
            assert controller.counters['peak_kv_bytes'] >= main_bytes + temporary_bytes
        monkeypatch.setattr(controller, '_observe_cache_peak', observe)
        qa.lm_head = RecordingHead((31,), fail_at=0 if fail else None)
        if fail:
            with pytest.raises(RuntimeError, match='controlled decode failure'):
                controller._generate_memory_text('Event notes:', visual_events=(_event(1, 4),))
        else:
            assert controller._generate_memory_text('Event notes:', visual_events=(_event(1, 4),)) == []
        assert observed and controller.counters['peak_kv_bytes'] >= max(observed) > main_bytes
        assert_main_untouched(controller, qa, before)
        assert not hasattr(controller, '_saved_summary_main_kv_bytes')
    finally:
        controller.close()
