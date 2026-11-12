"""CPU regressions for EventKV compensation, note rewriting, and text budgets.

The decoder tests use a real tiny Qwen model with a controlled output head.
They exercise actual KV appends/restoration rather than mocking generation.
No checkpoint, CUDA allocation, or benchmark is required.
"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
import pytest
import torch
_spec = importlib.util.spec_from_file_location('_eventkv_text_shared_fixtures', Path(__file__).with_name('test_initial_improvement_gpu.py'))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
_controller = _shared._controller
_event = _shared._event
_cache_copy = _shared._cache_copy

class _ControlledHead(torch.nn.Module):
    """Choose known token IDs while retaining all real decoder computation."""

    def __init__(self, tokens, *, fail_at=None):
        super().__init__()
        self.tokens = list(tokens)
        self.fail_at = fail_at
        self.calls = 0

    def forward(self, hidden):
        assert hidden.device.type == 'cpu'
        assert hidden.shape[-2:] == (1, 32)
        if self.calls == self.fail_at:
            raise RuntimeError('interrupted memory decoding')
        token = self.tokens[min(self.calls, len(self.tokens) - 1)]
        self.calls += 1
        logits = hidden.new_zeros((*hidden.shape[:-1], 32))
        logits[..., token] = 1
        return logits

def _snapshot(qa):
    return dict(cache=_cache_copy(qa), positions=[p.clone() for p in qa._position_ids_cache], offsets=qa._get_next_global_offset_per_layer())

def _assert_restored(qa, expected):
    assert len(qa.kv_cache) == len(expected['cache'])
    for actual, before in zip(qa.kv_cache, expected['cache']):
        for tensor, saved in zip(actual, before):
            torch.testing.assert_close(tensor, saved, atol=0, rtol=0)
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    for actual, before in zip(qa._position_ids_cache, expected['positions']):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
        assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    assert qa._get_next_global_offset_per_layer() == expected['offsets']
    assert qa._layer_position_ids == {}

@pytest.mark.parametrize('tokens,expected', [([31], []), ([7, 8, 31], [7, 8])])
def test_real_memory_generation_stops_at_eos_and_restores_visual_prompt_suffix(tokens, expected):
    controller, qa, _ = _controller(text_budget=2000)
    try:
        controller.retire_event(_event(0, 6))
        before = _snapshot(qa)
        qa.lm_head = head = _ControlledHead(tokens)
        result = controller._generate_memory_text('Additional event notes:', visual_events=(_event(1, 4), _event(2, 4)))
        assert result == expected
        assert head.calls == len(expected) + 1
        _assert_restored(qa, before)
        assert controller.counters['summary_append_calls'] == 1
        assert controller.counters['summary_append_generated_tokens'] == len(expected)
        assert controller.counters['summary_rewrite_calls'] == 0
    finally:
        controller.close()

@pytest.mark.parametrize('text_budget,max_new,kind,expected', [(2000, 64, 'append', 64), (2000, 64, 'rewrite', 64), (7, 64, 'rewrite', 7), (2000, 5, 'append', 5)])
def test_real_memory_generation_enforces_decode_limit_without_eos(text_budget, max_new, kind, expected):
    controller, qa, _ = _controller(text_budget=text_budget, summary_max_new_tokens=max_new)
    try:
        before = _snapshot(qa)
        qa.lm_head = head = _ControlledHead([7])
        result = controller._generate_memory_text('Event notes:', kind=kind)
        assert result == [7] * expected
        assert head.calls == expected
        _assert_restored(qa, before)
        assert controller.counters['summary_generation_calls'] == 1
        assert controller.counters['summary_generated_tokens'] == expected
        assert controller.counters[f'summary_{kind}_calls'] == 1
        assert controller.counters[f'summary_{kind}_generated_tokens'] == expected
        other = 'rewrite' if kind == 'append' else 'append'
        assert controller.counters[f'summary_{other}_calls'] == 0
        assert controller.counters[f'summary_{other}_seconds'] == 0
        assert controller.counters['summary_generation_seconds'] == controller.counters[f'summary_{kind}_seconds']
    finally:
        controller.close()

def test_memory_generation_stops_at_chat_end_token_too():
    controller, qa, _ = _controller(text_budget=2000)
    try:
        tokenizer = qa.processor.tokenizer
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda token: 30 if token == '<|im_end|>' else 0
        qa.lm_head = head = _ControlledHead([7, 30])
        before = _snapshot(qa)
        assert controller._generate_memory_text('Event notes:') == [7]
        assert head.calls == 2
        _assert_restored(qa, before)
    finally:
        controller.close()

def test_interrupted_memory_generation_restores_existing_event_kv_and_positions():
    controller, qa, _ = _controller(text_budget=2000)
    try:
        controller.retire_event(_event(0, 6))
        before = _snapshot(qa)
        qa.lm_head = head = _ControlledHead([7], fail_at=1)
        with pytest.raises(RuntimeError, match='interrupted memory decoding'):
            controller._generate_memory_text('Event notes:', visual_events=(_event(1, 4),))
        assert head.calls == 1
        _assert_restored(qa, before)
        assert controller._answer_lengths is None
        assert controller._capture_queries is False
        assert controller._capture_d is False
    finally:
        controller.close()

def test_compensation_prompt_adds_new_notes_without_question_or_old_note_fallback(monkeypatch):
    controller, qa, _ = _controller(text_budget=2000)
    try:
        left, right = (_event(0, 4), _event(1, 4))
        left.summary_ids, right.summary_ids = ([10, 11], [20, 21])
        qa.conv_history = ['SECRET_QUESTION_MUST_NOT_ENTER_MEMORY']
        calls = []

        def generate(prompt, *, visual_events=(), kind='append'):
            calls.append((prompt, visual_events, kind))
            return []
        monkeypatch.setattr(controller, '_generate_memory_text', generate)
        assert controller._make_summary(left, right, 0.8) == []
        assert left.summary_ids == [10, 11] and right.summary_ids == [20, 21]
        prompt, sources, kind = calls.pop()
        assert sources == (left, right) and kind == 'append'
        assert '10 11' in prompt and '20 21' in prompt
        assert 'additional' in prompt.lower()
        assert 'kept separately' in prompt and 'do not rewrite or repeat' in prompt
        assert 'SECRET_QUESTION' not in prompt
        assert list(inspect.signature(controller._make_summary).parameters) == ['left', 'right', 'cost']
    finally:
        controller.close()

def test_overlong_rewrite_uses_text_only_and_separate_real_generation_accounting(monkeypatch):
    controller, qa, _ = _controller(text_budget=2000)
    try:
        left, right = (_event(0, 4), _event(1, 4))
        source_ids = [4] * 101
        qa.lm_head = head = _ControlledHead([7, 8, 31])
        original_generate = controller._generate_memory_text
        calls = []

        def generate(prompt, *, visual_events=(), kind='append'):
            calls.append((prompt, visual_events, kind))
            return original_generate(prompt, visual_events=visual_events, kind=kind)
        monkeypatch.setattr(controller, '_generate_memory_text', generate)
        before = _snapshot(qa)
        result = controller._rewrite_summary(source_ids, left, right)
        assert result == [7, 8] and head.calls == 3
        assert source_ids == [4] * 101
        prompt, visual_events, kind = calls.pop()
        assert visual_events == () and kind == 'rewrite'
        assert 'Condense only' in prompt and 'Do not add information' in prompt
        assert qa.processor.tokenizer.decode(source_ids) in prompt
        assert controller.counters['summary_rewrite_calls'] == 1
        assert controller.counters['summary_append_calls'] == 0
        assert controller.counters['summary_rewrite_generated_tokens'] == 2
        assert controller.counters['summary_generation_seconds'] == controller.counters['summary_rewrite_seconds']
        _assert_restored(qa, before)
    finally:
        controller.close()

@pytest.mark.parametrize('count', [0, 30, 100])
def test_notes_at_or_below_event_limit_are_returned_without_generation(count, monkeypatch):
    controller, qa, _ = _controller(text_budget=2000)
    try:

        def forbidden(*args, **kwargs):
            raise AssertionError('notes within the event limit need no decoder call')
        monkeypatch.setattr(controller, '_generate_memory_text', forbidden)
        source = [4] * count
        result = controller._rewrite_summary(source, _event(0, 4), _event(1, 4))
        assert result == source
        assert result is not source
        assert controller.counters['summary_generation_calls'] == 0
    finally:
        controller.close()

def test_empty_rewrite_keeps_both_source_events_and_restores_the_cache(monkeypatch):
    controller, qa, _ = _controller(text_budget=2000)
    try:
        left, right = (_event(0, 4), _event(1, 4))
        left.summary_ids, right.summary_ids = ([4] * 50, [5] * 51)
        controller.memory.long_events = [left, right]
        monkeypatch.setattr(controller.memory, '_merge_costs', lambda: [0.1])
        qa.lm_head = _ControlledHead([31])
        before = _snapshot(qa)
        with pytest.raises(RuntimeError, match='rewrite produced no tokens'):
            controller.memory._merge_once()
        assert controller.memory.long_events[0] is left
        assert controller.memory.long_events[1] is right
        assert left.summary_ids == [4] * 50 and right.summary_ids == [5] * 51
        assert controller.memory.counters['merged_events'] == 0
        assert controller.counters['summary_rewrite_calls'] == 1
        _assert_restored(qa, before)
    finally:
        controller.close()

@pytest.mark.parametrize('combined_count', [100, 101])
def test_low_cost_visual_merge_applies_100_event_limit_with_2000_global_budget(combined_count, monkeypatch):
    controller, qa, _ = _controller(text_budget=2000)
    try:
        left, right = (_event(0, 4), _event(1, 4))
        left.summary_ids, right.summary_ids = ([4] * 50, [5] * (combined_count - 50))
        controller.memory.long_events = [left, right]
        monkeypatch.setattr(controller.memory, '_merge_costs', lambda: [0.1])
        qa.lm_head = head = _ControlledHead([7])
        before = _snapshot(qa)
        controller.memory._merge_once()
        controller._enforce_text_budget()
        assert len(controller.memory.long_events) == 1
        merged = controller.memory.long_events[0]
        assert merged.token_count == 4
        assert controller.counters['summary_append_calls'] == 0
        assert controller.memory.counters['summary_requests'] == 0
        if combined_count == 100:
            assert merged.summary_ids == [4] * 50 + [5] * 50
            assert head.calls == 0
            assert controller.counters['summary_rewrite_calls'] == 0
            assert controller.memory.counters['text_rewrite_requests'] == 0
        else:
            assert merged.summary_ids == [7] * 64
            assert head.calls == 64
            assert controller.counters['summary_rewrite_calls'] == 1
            assert controller.memory.counters['text_rewrite_requests'] == 1
        assert len(merged.summary_ids) <= 100
        _assert_restored(qa, before)
    finally:
        controller.close()

def test_global_2000_allows_multiple_notes_then_fifo_trims_matching_scoring_queries():
    controller, qa, config = _controller(text_budget=2000)
    try:
        for event_id in range(2):
            source = _event(event_id, 4)
            source.summary_ids = [7 + event_id] * 64
            controller.retire_event(source)
        latest = _event(2, 4)
        latest.summary_ids = [9] * 64
        controller.memory.long_events.append(latest)
        first, second = controller.kv_events
        first_text_q, first_positions = (first.query.clone(), first.query_positions.clone())
        second_text_q, second_positions = (second.query.clone(), second.query_positions.clone())
        before = _snapshot(qa)
        controller._enforce_text_budget()
        assert [len(entry.summary_ids) for entry in (first, second, latest)] == [64, 64, 64]
        assert controller.get_stats()['text_tokens'] == 192
        config.text_budget = 150
        controller._enforce_text_budget()
        assert [len(entry.summary_ids) for entry in (first, second, latest)] == [22, 64, 64]
        assert first.summary_ids == [7] * 22
        torch.testing.assert_close(first.query, first_text_q[:, :, -22:], atol=0, rtol=0)
        torch.testing.assert_close(first.query_positions, first_positions[-22:], atol=0, rtol=0)
        assert first.query_source == 'text'
        assert first.query.untyped_storage().nbytes() == first.query.numel() * first.query.element_size()
        torch.testing.assert_close(second.query, second_text_q, atol=0, rtol=0)
        config.text_budget = 100
        controller._enforce_text_budget()
        assert [len(entry.summary_ids) for entry in (first, second, latest)] == [0, 36, 64]
        assert first.query_source == 'visual'
        torch.testing.assert_close(first.query, first.visual_query, atol=0, rtol=0)
        torch.testing.assert_close(first.query_positions, first.visual_query_positions, atol=0, rtol=0)
        assert second.query_source == 'text'
        torch.testing.assert_close(second.query, second_text_q[:, :, -36:], atol=0, rtol=0)
        torch.testing.assert_close(second.query_positions, second_positions[-36:], atol=0, rtol=0)
        assert controller.counters['summary_generation_calls'] == 0
        _assert_restored(qa, before)
        config.kv_budget = 4
        controller.compact_kv()
        assert all((state.score_builds == 1 for state in controller.kv_events))
        assert all((torch.isfinite(scores).all() for state in controller.kv_events for scores in state.scores_by_layer))
        assert controller.get_stats()['query_cache_bytes'] == 0
    finally:
        controller.close()
