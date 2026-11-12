"""Behavioral contracts for the EventKV visual tiers; no model/GPU required."""
from types import SimpleNamespace
import weakref
import pytest
import torch
from versions.latest.eventkv_memory import EventVectorMemory, VisualEvent, VisualFrame, merge_half

def config(**overrides):
    values = dict(recent_budget=4, recent_fallback_budget=400, event_budget=200, event_window=8, min_event_frames=2, short_merge_rounds=2, similarity_weight=0.4, time_weight=0.2, merge_weight=0.4, summary_cost_threshold=0.5, event_text_max_tokens=100, summary_max_new_tokens=64, retire_policy='overflow', retire_merge_depth=5)
    values.update(overrides)
    return SimpleNamespace(**values)

def frame(index, count=4, direction=(1.0, 0.0)):
    features = torch.tensor(direction).expand(count, -1).clone()
    positions = torch.stack((torch.full((count,), float(index)), torch.arange(count).float(), torch.zeros(count)))
    return VisualFrame(features, positions, torch.full((count,), index / 100.0), index, float(index))

def event(index, count=16):
    f = frame(index, count)
    return VisualEvent(index, f.features, f.positions, f.novelty, torch.ones(count), float(index), float(index), frame_count=2)

def test_weighted_half_merge_preserves_mass_novelty_and_representative_positions():
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    positions = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    novelty = torch.tensor([1.0, 0.0, 0.2, 0.4])
    weights = torch.tensor([2.0, 1.0, 1.0, 3.0])
    x, p, n, w = merge_half(values, positions, novelty, weights)
    torch.testing.assert_close(x, torch.eye(2))
    torch.testing.assert_close(p, positions[:, [1, 3]])
    torch.testing.assert_close(n, torch.tensor([2 / 3, 0.35]))
    torch.testing.assert_close(w, torch.tensor([3.0, 4.0]))
    x2, p2, n2, w2 = merge_half(x, p, n, w)
    torch.testing.assert_close(x2, torch.tensor([[3 / 7, 4 / 7]]))
    torch.testing.assert_close(n2, torch.tensor([3.4 / 7]))
    torch.testing.assert_close(w2, torch.tensor([7.0]))
    torch.testing.assert_close(p2, positions[:, [3]])

@pytest.mark.parametrize('count,expected', [(1, 1), (2, 1), (5, 2), (8, 2), (9, 3)])
def test_two_half_merges_round_up_to_a_quarter_without_nan(count, expected):
    f = frame(0, count, direction=(0.0, 0.0))
    x, p, n, w = (f.features, f.positions, f.novelty, None)
    for _ in range(2):
        x, p, n, w = merge_half(x, p, n, w)
    assert len(x) == expected
    assert torch.isfinite(x).all() and torch.isfinite(n).all()
    assert w.sum().item() == count
    assert p.shape == (3, expected)

def test_short_queue_seals_at_eight_without_eighteen_extra_frames():
    memory = EventVectorMemory(config(), on_retire=lambda _: None)
    for index in range(8):
        memory.add_frame(frame(index))
        assert len(memory.short) < 8
    assert memory.counters['formed_events'] == 0
    assert len(memory.short) == 7
    memory.add_frame(frame(8))
    assert memory.counters['formed_events'] == 1
    assert len(memory.short) == 6
    assert [f.frame_id for f in memory.recent] == [8]
    first = memory.long_events[0]
    assert first.frame_count == 2
    assert (first.start_time, first.end_time) == (0.0, 1.0)
    assert first.token_count == 1
    assert first.weights.sum().item() == 8
    assert memory.get_stats()['peak_short_frames'] == 8

def test_boundary_uses_original_frame_changes_and_excludes_one_frame_event():
    memory = EventVectorMemory(config(), on_retire=lambda _: None)
    directions = [(1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    for index, direction in enumerate(directions):
        memory.add_frame(frame(index, direction=direction))
    first = memory.long_events[0]
    assert first.frame_count == 3
    assert first.end_time == 2.0
    assert len(memory.short) == 5

def test_recent_uses_authorized_fallback_and_preserves_complete_frames():
    memory = EventVectorMemory(config(recent_budget=50), on_retire=lambda _: None)
    for index in range(3):
        memory.add_frame(frame(index, count=144))
    assert memory.effective_recent_budget == 400
    assert memory.counters['recent_fallbacks'] == 1
    assert [f.frame_id for f in memory.recent] == [1, 2]
    assert memory.recent_tokens == 288
    assert memory.short[0].token_count == 36
    assert memory.short[0].weights.sum().item() == 144
    with pytest.raises(ValueError, match='exceeding fallback'):
        memory.add_frame(frame(3, count=401))
    assert memory.counters['frames_seen'] == 3

def test_long_overflow_alternates_merge_and_retire_until_within_budget():
    retired = []
    summaries = []

    def retire(value):
        retired.append((value.event_id, value.token_count, value.start_time, value.end_time))

    def summarize(left, right, cost):
        summaries.append((left.event_id, right.event_id, left.token_count, right.token_count))
        return [99, right.event_id]
    memory = EventVectorMemory(config(event_budget=20, summary_cost_threshold=0.0), on_retire=retire, on_summary=summarize)
    memory.current_time = 7.0
    memory.long_events = [event(i) for i in range(6)]
    for item in memory.long_events:
        item.summary_ids = [item.event_id]
    memory._enforce_event_budget()
    assert [op['kind'] for op in memory.operation_log] == ['merge', 'retire', 'merge', 'retire', 'merge']
    assert [item[0] for item in retired] == [0, 2]
    assert len(memory.long_events) == 1 and memory.event_tokens == 16
    assert memory.long_events[0].summary_ids == [4, 5, 99, 5]
    assert summaries == [(0, 1, 16, 16), (2, 3, 16, 16), (4, 5, 16, 16)]
    assert memory.counters['summary_requests'] == 3
    assert memory.counters['retired_events'] == 2

def test_binary_event_merge_tracks_round_depth_separately_from_initial_count():
    memory = EventVectorMemory(config())
    memory._merge_costs = lambda: [0.0] * (len(memory.long_events) - 1)
    memory.long_events = [event(0), event(1)]
    memory._merge_once()
    left = memory.long_events[0]
    assert (left.merge_count, left.merge_depth) == (2, 1)
    memory.long_events = [event(2), event(3)]
    memory._merge_once()
    right = memory.long_events[0]
    memory.long_events = [left, right]
    memory._merge_once()
    balanced = memory.long_events[0]
    assert (balanced.merge_count, balanced.merge_depth) == (4, 2)

def test_fifth_round_retires_oldest_even_when_merge_already_fits_budget():
    retired = []
    memory = EventVectorMemory(config(event_budget=10, retire_policy='merge_depth'), on_retire=lambda item: retired.append(item))
    memory._merge_costs = lambda: [0.0]
    left, right = (event(0, 8), event(1, 8))
    left.merge_count, left.merge_depth = (5, 4)
    right.merge_count, right.merge_depth = (3, 2)
    memory.long_events = [left, right]
    memory._enforce_event_budget()
    assert [(e.event_id, e.token_count, e.merge_count, e.merge_depth) for e in retired] == [(0, 8, 8, 5)]
    assert not memory.long_events
    assert memory.counters['retired_by_merge_depth'] == 1
    assert memory.operation_log[-1]['reason'] == 'merge_depth'

def test_remaining_overflow_does_not_retire_immature_oldest():
    retired = []
    memory = EventVectorMemory(config(event_budget=20, retire_policy='merge_depth'), on_retire=lambda item: retired.append(item))
    memory._merge_costs = lambda: [0.0] * (len(memory.long_events) - 1)
    memory.long_events = [event(i) for i in range(4)]
    memory._enforce_event_budget()
    assert retired == []
    assert memory.event_tokens == 16
    assert memory.counters['merged_events'] == 3
    assert memory.long_events[0].merge_count == 4
    assert memory.long_events[0].merge_depth == 3
    assert [op['kind'] for op in memory.operation_log] == ['merge'] * 3

def test_mature_newer_event_waits_until_it_is_oldest():
    retired = []
    memory = EventVectorMemory(config(event_budget=18, retire_policy='merge_depth'), on_retire=lambda item: retired.append(item.event_id))
    memory._merge_costs = lambda: [1.0, 0.0]
    oldest, newer, newest = [event(i, 8) for i in range(3)]
    oldest.merge_count, oldest.merge_depth = (2, 1)
    newer.merge_count, newer.merge_depth = (5, 4)
    memory.long_events = [oldest, newer, newest]
    memory._enforce_event_budget()
    assert retired == []
    assert [e.event_id for e in memory.long_events] == [0, 1]
    assert [e.merge_depth for e in memory.long_events] == [1, 5]
    assert memory.event_tokens == 16

def test_already_mature_successor_retires_when_it_becomes_oldest():
    retired = []
    memory = EventVectorMemory(config(event_budget=20, retire_policy='merge_depth'), on_retire=lambda item: retired.append((item.event_id, item.merge_depth)))
    memory._merge_costs = lambda: [0.0, 1.0]
    first, second, third = [event(i, 8) for i in range(3)]
    first.merge_count, first.merge_depth = (5, 4)
    third.merge_count, third.merge_depth = (8, 6)
    memory.long_events = [first, second, third]
    memory._enforce_event_budget()
    assert retired == [(0, 5), (2, 6)]
    assert not memory.long_events
    assert memory.counters['retired_by_merge_depth'] == 2

def test_single_oversized_immature_event_is_compacted_without_fake_event_rounds():
    retired = []
    memory = EventVectorMemory(config(event_budget=2, retire_policy='merge_depth'), on_retire=lambda item: retired.append(item))
    sole = event(0, 9)
    memory.long_events = [sole]
    memory._enforce_event_budget()
    assert retired == []
    assert memory.event_tokens == 2
    assert sole.merge_count == 1 and sole.merge_depth == 0
    assert sole.weights.sum().item() == 9
    assert memory.counters['merged_events'] == 0
    assert memory.counters['single_event_compactions'] == 3
    assert all((op['kind'] == 'compact_single' for op in memory.operation_log))

def test_merge_depth_does_not_replace_count_in_streamforest_cost():
    memory = EventVectorMemory(config())
    memory.current_time = 10.0
    memory.long_events = [event(i) for i in range(3)]
    oldest = memory.long_events[0]
    oldest.merge_count, oldest.merge_depth = (8, 3)
    memory.long_events[1].merge_count = 2
    memory.long_events[1].merge_depth = 1
    before = memory._merge_costs()
    oldest.merge_depth = 7
    assert memory._merge_costs() == before
    assert memory.get_stats()['long_event_initial_counts'] == [8, 2, 1]
    assert memory.get_stats()['long_event_merge_depths'] == [7, 1, 0]

@pytest.mark.parametrize('total_text_tokens', [30, 100])
def test_low_cost_merge_concatenates_notes_without_generation_or_rewrite(total_text_tokens):

    def forbidden(*args):
        raise AssertionError('text within the event limit must not be generated or rewritten')
    cfg = config()
    del cfg.event_text_max_tokens
    memory = EventVectorMemory(cfg, on_summary=forbidden, on_rewrite=forbidden)
    memory._merge_costs = lambda: [0.1]
    left, right = (event(0), event(1))
    left.summary_ids = list(range(total_text_tokens // 2))
    right.summary_ids = list(range(total_text_tokens // 2, total_text_tokens))
    memory.long_events = [left, right]
    memory._merge_once()
    assert memory.long_events[0].summary_ids == list(range(total_text_tokens))
    assert memory.counters['summary_requests'] == 0
    assert memory.counters['text_rewrite_requests'] == 0
    operation = memory.operation_log[-1]
    assert operation['text_tokens_before'] == total_text_tokens
    assert operation['text_tokens_generated'] == 0
    assert operation['text_tokens_after'] == total_text_tokens
    assert operation['text_rewritten'] is False

def test_high_cost_compensation_is_appended_after_both_existing_notes():
    left, right = (event(0), event(1))
    left.summary_ids, right.summary_ids = ([10, 11], [20, 21])

    def compensate(source_left, source_right, cost):
        assert source_left is left and source_right is right
        assert cost == 0.8
        return [30, 31]
    memory = EventVectorMemory(config(), on_summary=compensate)
    memory._merge_costs = lambda: [0.8]
    memory.long_events = [left, right]
    memory._merge_once()
    assert memory.long_events[0].summary_ids == [10, 11, 20, 21, 30, 31]
    assert left.summary_ids == [10, 11] and right.summary_ids == [20, 21]
    assert memory.counters['summary_requests'] == 1
    assert memory.counters['text_rewrite_requests'] == 0
    operation = memory.operation_log[-1]
    assert operation['summary_requested'] is True
    assert operation['text_tokens_before'] == 4
    assert operation['text_tokens_generated'] == 2
    assert operation['text_tokens_after'] == 6
    assert operation['text_rewritten'] is False

def test_low_cost_notes_over_100_are_rewritten_without_visual_compensation():
    left, right = (event(0), event(1))
    left.summary_ids, right.summary_ids = (list(range(50)), list(range(50, 101)))
    rewritten = []

    def rewrite(combined_ids, source_left, source_right):
        assert source_left is left and source_right is right
        rewritten.append(list(combined_ids))
        return [700] * 64

    def forbidden(*args):
        raise AssertionError('low merge cost must not request visual compensation')
    memory = EventVectorMemory(config(), on_summary=forbidden, on_rewrite=rewrite)
    memory._merge_costs = lambda: [0.1]
    memory.long_events = [left, right]
    memory._merge_once()
    assert rewritten == [list(range(101))]
    assert memory.long_events[0].summary_ids == [700] * 64
    assert left.summary_ids == list(range(50))
    assert right.summary_ids == list(range(50, 101))
    assert memory.counters['summary_requests'] == 0
    assert memory.counters['text_rewrite_requests'] == 1
    operation = memory.operation_log[-1]
    assert operation['summary_requested'] is False
    assert operation['text_tokens_before'] == 101
    assert operation['text_tokens_generated'] == 0
    assert operation['text_tokens_after'] == 64
    assert operation['text_rewritten'] is True

def test_compensation_is_appended_before_checking_the_event_text_limit():
    left, right = (event(0), event(1))
    left.summary_ids, right.summary_ids = (list(range(49)), list(range(49, 98)))
    calls = []

    def compensate(*args):
        calls.append('compensate')
        return [98, 99, 100]

    def rewrite(combined_ids, source_left, source_right):
        assert combined_ids == list(range(101))
        assert source_left is left and source_right is right
        calls.append('rewrite')
        return [900] * 64
    memory = EventVectorMemory(config(), on_summary=compensate, on_rewrite=rewrite)
    memory._merge_costs = lambda: [0.8]
    memory.long_events = [left, right]
    memory._merge_once()
    assert calls == ['compensate', 'rewrite']
    assert memory.long_events[0].summary_ids == [900] * 64
    assert memory.counters['summary_requests'] == 1
    assert memory.counters['text_rewrite_requests'] == 1
    operation = memory.operation_log[-1]
    assert operation['text_tokens_before'] == 98
    assert operation['text_tokens_generated'] == 3
    assert operation['text_tokens_after'] == 64
    assert operation['text_rewritten'] is True

def test_overlong_event_text_requires_a_rewrite_callback_before_visual_merge():
    memory = EventVectorMemory(config())
    memory._merge_costs = lambda: [0.1]
    left, right = (event(0), event(1))
    left.summary_ids, right.summary_ids = ([1] * 50, [2] * 51)
    memory.long_events = [left, right]
    with pytest.raises(RuntimeError, match='on_rewrite is not configured'):
        memory._merge_once()
    assert memory.long_events == [left, right]
    assert memory.counters['merged_events'] == 0
    assert not memory.operation_log

@pytest.mark.parametrize('event_limit,generation_limit,returned', [(100, 64, 65), (6, 64, 7)])
def test_rewrite_callback_must_respect_both_token_limits(event_limit, generation_limit, returned):
    memory = EventVectorMemory(config(event_text_max_tokens=event_limit, summary_max_new_tokens=generation_limit), on_rewrite=lambda *args: [7] * returned)
    memory._merge_costs = lambda: [0.1]
    left, right = (event(0), event(1))
    left.summary_ids, right.summary_ids = ([1] * event_limit, [2])
    memory.long_events = [left, right]
    with pytest.raises(ValueError, match='on_rewrite returned.*exceeding limit'):
        memory._merge_once()
    assert memory.long_events == [left, right]
    assert memory.counters['merged_events'] == 0
    assert not memory.operation_log

def test_an_oversized_single_event_retires_without_an_infinite_merge_loop():
    retired = []
    memory = EventVectorMemory(config(event_budget=2), on_retire=lambda value: retired.append(value.event_id))
    memory.long_events = [event(3, 9)]
    memory._enforce_event_budget()
    assert retired == [3] and memory.event_tokens == 0

def test_question_flush_seals_a_single_short_tail_and_leaves_recent_raw():
    memory = EventVectorMemory(config(), on_retire=lambda _: None)
    memory.add_frame(frame(0))
    memory.add_frame(frame(1))
    memory.flush_short()
    assert not memory.short
    assert memory.long_events[0].frame_count == 1
    assert memory.long_events[0].weights.sum().item() == 4
    assert memory.recent[0].token_count == 4
    count = memory.counters['formed_events']
    memory.flush_short()
    assert memory.counters['formed_events'] == count

def test_original_similarity_survives_variable_grid_frame_retirement():
    memory = EventVectorMemory(config(recent_fallback_budget=8), on_retire=lambda _: None)
    memory.add_frame(frame(0, count=4, direction=(1.0, 0.0)))
    memory.add_frame(frame(1, count=6, direction=(0.0, 1.0)))
    memory.add_frame(frame(2, count=3, direction=(1.0, 0.0)))
    assert [entry.frame_id for entry in memory.short] == [0, 1]
    assert memory.short[1].token_count == 2
    assert memory.short[1].similarity_from_previous == pytest.approx(0.0)
    assert memory.recent[0].similarity_from_previous == pytest.approx(0.0)

def test_sealed_short_payload_is_released_before_retirement_prefill_callback():
    observed = []
    references = []

    def retire(value):
        observed.append(value.event_id)
        assert all((reference() is None for reference in references))
    memory = EventVectorMemory(config(event_budget=0), on_retire=retire)
    memory.short = [frame(0), frame(1)]
    references.extend((weakref.ref(entry.features) for entry in memory.short))
    memory._form_event(2)
    assert observed == [0]
    assert not memory.short and (not memory.long_events)

def test_kept_frames_and_singleton_merges_do_not_pin_deleted_batch_storage(monkeypatch):

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError('visual payload must not be copied to CPU')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden_cpu)
    storage = torch.ones(1000, 2)
    f = frame(0)
    f.features = storage[:4]
    memory = EventVectorMemory(config(), on_retire=lambda _: None)
    memory.add_frame(f)
    kept = memory.recent[0].features
    assert kept.untyped_storage().nbytes() == kept.numel() * kept.element_size()
    assert kept.data_ptr() != storage.data_ptr()
    p = torch.zeros(3, 1000)
    x, positions, n, w = merge_half(storage[:1], p[:, :1], torch.zeros(1))
    for tensor in (x, positions, n, w):
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    memory.add_frame(frame(1))
    assert all((value.features.device == kept.device for value in memory.recent + memory.short))

def test_reset_drops_all_tiers_and_stream_order_is_validated():
    memory = EventVectorMemory(config(), on_retire=lambda _: None)
    for index in range(12):
        memory.add_frame(frame(index))
    with pytest.raises(ValueError, match='strictly increasing'):
        memory.add_frame(frame(10))
    memory.reset()
    assert not memory.recent and (not memory.short) and (not memory.long_events)
    assert memory.get_stats()['vector_resident_bytes'] == 0
    assert memory.counters['frames_seen'] == 0
    memory.add_frame(frame(0))
