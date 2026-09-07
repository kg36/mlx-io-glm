"""Tiny native cache transactions; no model, real SSD reads, or perf runs."""
import mlx.core as mx
import pytest


def _cache(kind):
    if kind == "global":
        state = mx._expert_ssd_global_pool_state_new(
            16, [3], [6], 3, .9, .25, 32., .25, 0., 300., 3, .05, 4.,
        )

        def plan(route, mask=None):
            return mx._expert_ssd_global_pool_plan(
                state, 3, mx.array(route, dtype=mx.int32),
                **({"active_mask": mask} if mask is not None else {}),
            )

        metadata = lambda: mx._expert_ssd_global_pool_metadata(state)
        snapshot = lambda: mx._expert_ssd_global_pool_policy_snapshot(state)
    else:
        markov = mx._expert_ssd_markov_state_new(16, 3)
        state = mx._expert_ssd_route_cache_state_new(
            16, 9, .9, .25, 32., .25, 300., markov,
        )

        def plan(route, mask=None):
            return mx._expert_ssd_route_cache_plan(
                state, mx.array(route, dtype=mx.int32),
                **({"active_mask": mask} if mask is not None else {}),
            )

        metadata = lambda: mx._expert_ssd_route_cache_metadata(state)
        snapshot = lambda: mx._expert_ssd_markov_snapshot(markov)
    plan(list(range(9)))
    return plan, metadata, snapshot


@pytest.mark.parametrize("kind", ["private", "global"])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_masked_filler_policy_equals_actual_demand(kind, width):
    candidate, candidate_meta, candidate_policy = _cache(kind)
    control, control_meta, control_policy = _cache(kind)
    # 7 is a shared filler in the global pool. 6 is a filler on one row,
    # but genuinely used later; first-active ordering/count must be honored.
    route = ([0, 6, 1, 2, 7, 6] * width)
    mask = ([1, 0, 1, 1, 0, 1] * width)
    actual = candidate(route, mask)
    expected = control([e for e, active in zip(route, mask) if active])
    assert actual["routed"] == route  # Full compute geometry retained.
    assert actual["counts"][7] == 0
    assert actual["counts"][6] == width
    assert actual["hits"] == expected["hits"] == 4
    assert actual["misses"] == expected["misses"] == 0
    assert candidate_meta() == control_meta()
    assert candidate_policy() == control_policy()


@pytest.mark.parametrize("kind", ["private", "global"])
def test_mask_pins_cold_policy_filler_during_eviction(kind):
    plan, metadata, _ = _cache(kind)
    before = metadata()
    # Protect all nine physical rows except 8, but only touch 0..6 and load9.
    # Filler7 is older than most demand rows; it must still not be a victim.
    actual = plan([0, 1, 2, 3, 4, 5, 6, 7, 9], [1] * 7 + [0, 1])
    assert actual["missing"] == [9]
    assert actual["counts"][7] == 0
    assert actual["hits"] == 7
    assert actual["evicted"] == ([(3, 8)] if kind == "global" else [8])
    after = metadata()
    old_layer = before["layers"][0] if kind == "global" else before
    new_layer = after["layers"][0] if kind == "global" else after
    assert new_layer["last_seen"][7] == old_layer["last_seen"][7]
    assert new_layer["demand"][7] == old_layer["demand"][7]


@pytest.mark.parametrize("kind", ["private", "global"])
@pytest.mark.parametrize("route,mask", [([0, 1], [1]), ([0, 10], [1, 0]), ([0], [0])])
def test_bad_mask_fails_before_mutating_cache(kind, route, mask):
    plan, metadata, snapshot = _cache(kind)
    before, policy = metadata(), snapshot()
    with pytest.raises(ValueError):
        plan(route, mask)
    assert metadata() == before
    assert snapshot() == policy
