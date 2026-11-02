"""The check that sizes a depth pass to the machine it is on. No model, no torch work."""

from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from depthc import memory
from depthc.video_io import MATTE_SAMPLE_FRAMES

GB = memory.GB
CLIP_720P_8S = (192, 720, 1280, 3)

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              100000.
Pages active:                            500000.
Pages inactive:                           50000.
Pages speculative:                        10000.
Pages throttled:                              0.
Pages wired down:                        200000.
"""


def _machine(monkeypatch, total_gb: float, free_gb: float | None, own_gb: float = 0) -> None:
    monkeypatch.delenv(memory.OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(memory, "own_bytes", lambda: int(own_gb * GB))
    monkeypatch.setattr(memory, "total_bytes", lambda: int(total_gb * GB))
    monkeypatch.setattr(
        memory, "available_bytes", lambda: None if free_gb is None else int(free_gb * GB)
    )


def test_vm_stat_counts_free_inactive_and_speculative_pages():
    assert memory.parse_vm_stat(VM_STAT) == (100000 + 50000 + 10000) * 16384


@pytest.mark.parametrize(
    "broken",
    [
        "",
        "not vm_stat at all",
        VM_STAT.replace("page size of 16384 bytes", "page size unknown"),
        VM_STAT.replace("Pages inactive", "Pages elsewhere"),
    ],
)
def test_vm_stat_that_cannot_be_read_says_so(broken):
    assert memory.parse_vm_stat(broken) is None


def test_this_machine_reads_real_numbers():
    """A real `vm_stat` must parse, or the check quietly falls back to half of RAM everywhere."""
    free = memory.available_bytes()
    if shutil.which("vm_stat"):
        assert free is not None and 0 < free <= memory.total_bytes()
    else:  # the Linux CI runner: no vm_stat, so the fallback is what runs there
        assert free is None


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("vm_stat", 5)])
def test_missing_vm_stat_reads_as_unknown(monkeypatch, error):
    def fails(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(memory.subprocess, "run", fails)
    assert memory.available_bytes() is None


@pytest.mark.parametrize("input_size, measured_gb", [(518, 7.3), (392, 5.0)])
def test_estimate_sits_just_above_what_was_measured(input_size, measured_gb):
    """Under the measurement invites the stall this exists to stop. Far over it refuses passes
    that fit, and a check people keep having to override stops protecting them."""
    assert measured_gb * GB <= memory.need_bytes(CLIP_720P_8S, input_size) <= measured_gb * 1.25 * GB


@pytest.mark.parametrize(
    "input_size, smallest_limit_that_finished",
    [(140, 1.32), (196, 1.29), (252, 2.39), (280, 2.39), (336, 2.74), (406, 3.50), (462, 4.68), (518, 5.55)],
)
def test_pool_covers_every_measured_size(input_size, smallest_limit_that_finished):
    """Scaling one measurement by area alone promised 1.8 GB at 280, which ran out on a real pass.
    The refusal's suggestion stands on this estimate, so it has to hold at small sizes too."""
    assert memory._pool_need(720, 1280, input_size) >= smallest_limit_that_finished * GB


def test_the_end_of_the_pass_is_counted():
    """Upstream's alignment holds extra copies of the depth after the windows finish. At a small
    picture size that is the peak: 8 s of 720 x 1280 at 140 read 5.1 GB, with the windows at 2.9."""
    assert memory.need_bytes((192, 1280, 720, 3), 140) >= 5.1 * GB


def test_a_long_clip_is_sized_by_its_copies_of_the_depth():
    """30 s of 720p. Counting one copy of the depth would come out more than 2 GB short."""
    long_clip = (720, 720, 1280, 3)
    one_copy = memory._pool_need(720, 1280, 518) + memory._PROCESS + 720 * 720 * 1280 * 7
    assert memory.need_bytes(long_clip, 518) >= one_copy + 2 * GB


def test_a_short_clip_still_costs_whole_windows():
    """Upstream pads to 32 frames and keeps them all. Two frames of 8K cost what 32 do, at 4 bytes
    a pixel, twice over, which counting the clip's own two frames would miss by about 8 GB."""
    pixels = 7680 * 4320
    assert memory.need_bytes((2, 4320, 7680, 3), 140) >= pixels * memory._DEPTH * memory._WINDOW * 2


def test_a_clip_larger_than_was_measured_adds_its_upsampled_depth():
    same_shape = memory._pool_need(2160, 3840, 518) - memory._pool_need(720, 1280, 518)
    assert same_shape == (3840 * 2160 - 1280 * 720) * memory._WINDOW * memory._DEPTH
    assert same_shape > 0.85 * GB


def test_a_smaller_picture_cannot_shrink_what_the_clip_itself_holds():
    """Past a point only a shorter clip helps, so no --input-size may be offered as if it would."""
    floor = memory.need_bytes(CLIP_720P_8S, 140)
    assert memory.largest_input_size_that_fits(CLIP_720P_8S, 518, floor - 1) is None
    assert memory.largest_input_size_that_fits(CLIP_720P_8S, 518, floor) is not None


@pytest.mark.parametrize(
    "total, free, fits",
    [
        (8, 5, False),  # the small laptop
        (16, 6, False),
        (16, 12, True),
        (24, 3, False),  # the Mac that stalled, with everything open
        (24, 12, True),
        (64, 40, True),
    ],
)
def test_the_answer_comes_from_the_machine(monkeypatch, total, free, fits):
    _machine(monkeypatch, total, free)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert (plan.refusal is None) is fits
    assert plan.allowed == int(free * GB) - memory._KEEP_FREE


def test_refusal_is_plain_and_names_the_figures(monkeypatch):
    _machine(monkeypatch, 24, 3)
    refusal = memory.plan(CLIP_720P_8S, 518).refusal
    assert refusal is not None
    assert "3.0 GB free out of 24.0 GB" in refusal
    assert f"about {memory.need_bytes(CLIP_720P_8S, 518) / GB:.1f} GB" in refusal
    assert "Close other apps" in refusal
    assert memory.OVERRIDE_ENV in refusal


def test_refusal_offers_a_smaller_size_that_really_fits(monkeypatch):
    _machine(monkeypatch, 16, 8)
    plan = memory.plan(CLIP_720P_8S, 518)
    smaller = memory.largest_input_size_that_fits(CLIP_720P_8S, 518, plan.allowed)
    assert smaller is not None and smaller % 14 == 0
    assert memory.need_bytes(CLIP_720P_8S, smaller) <= plan.allowed
    assert memory.need_bytes(CLIP_720P_8S, smaller + 14) > plan.allowed  # the largest, not just one
    assert f"--input-size {smaller}" in plan.refusal
    assert "shorter clip" not in plan.refusal
    # What was offered has to pass when it is tried.
    assert memory.plan(CLIP_720P_8S, smaller).refusal is None


def test_refusal_offers_no_size_when_none_fits(monkeypatch):
    _machine(monkeypatch, 8, 2)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.refusal is not None
    assert "--input-size" not in plan.refusal
    assert "Close other apps" in plan.refusal
    assert "shorter clip" in plan.refusal


def test_what_this_process_holds_is_not_charged_twice(monkeypatch):
    """A second pass in one run was refused at 5.0 GB against 5.4 after the first fit at 6.2:
    the first had left 1.4 GB in the process, which `vm_stat` no longer counted as free."""
    free = memory.need_bytes(CLIP_720P_8S, 518) / GB + memory._KEEP_FREE / GB - 1.3
    _machine(monkeypatch, 24, free)
    assert memory.plan(CLIP_720P_8S, 518).refusal is not None
    _machine(monkeypatch, 24, free, own_gb=1.4)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.refusal is None
    assert plan.allowed == int(free * GB) + int(1.4 * GB) - memory._KEEP_FREE


def test_this_process_can_read_what_it_holds():
    """The real call on a real allocation, because a wrong offset into the system's struct would
    read as a plausible number. On the Linux CI runner the call does not exist, so it says 0."""
    before = memory.own_bytes()
    if sys.platform != "darwin":
        assert before == 0
        return
    assert 0 < before < memory.total_bytes()
    ballast = np.ones(300 * 1024**2, dtype=np.uint8)
    assert memory.own_bytes() - before > 200 * 1024**2
    del ballast


def test_a_clip_too_big_to_read_is_refused_before_it_is_read(monkeypatch):
    """Reading holds the frames twice for a moment, so 30 s of source-resolution 4K wants 33 GB."""
    frames = 720 * 3840 * 2160 * 3
    _machine(monkeypatch, 24, 20)
    refusal = memory.decode_refusal(frames)
    assert refusal is not None and "to read this clip" in refusal and "Nothing ran" in refusal
    assert memory.OVERRIDE_ENV in refusal
    # Twice the frames, and the margin, out of what is free. Not one copy, and not without the margin.
    edge = 2 * frames + memory._KEEP_FREE
    _machine(monkeypatch, 64, edge / GB + 0.01)
    assert memory.decode_refusal(frames) is None
    _machine(monkeypatch, 64, edge / GB - 0.01)
    assert memory.decode_refusal(frames) is not None


def test_a_read_does_not_count_what_this_process_holds_as_room(monkeypatch):
    """`look-compare --manifest` still holds the last clip's frames while it reads the next, so a
    credit would approve a read that runs the machine out."""
    frames = 4 * GB
    _machine(monkeypatch, 24, 6, own_gb=4.5)
    assert memory.decode_refusal(frames) is not None


def test_a_small_read_is_never_refused(monkeypatch):
    """The matte scan runs on every clip, on surfaces with no handler for this, and `inbox --no-scout`
    and the UI's readout should survive a tight machine. Nine frames of 8K stay under the line."""
    _machine(monkeypatch, 24, 0.5)
    assert memory.decode_refusal(MATTE_SAMPLE_FRAMES * 7680 * 4320 * 3) is None
    assert memory.decode_refusal(memory._SMALL_READ - 1) is None
    assert memory.decode_refusal(memory._SMALL_READ) is not None


def test_a_hand_set_limit_is_for_the_whole_pass_so_it_gets_no_credit(monkeypatch):
    _machine(monkeypatch, 24, 9.1, own_gb=1)
    monkeypatch.setenv(memory.OVERRIDE_ENV, "8.5")
    assert memory.plan(CLIP_720P_8S, 518).allowed == int(8.5 * GB)


def test_limit_leaves_the_pass_what_it_needs_and_no_more_than_is_allowed(monkeypatch):
    _machine(monkeypatch, 24, 12)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.gpu_limit >= memory._pool_need(720, 1280, 518)
    assert plan.gpu_limit < plan.allowed


def test_limit_at_the_edge_of_fitting_still_covers_the_pool(monkeypatch):
    need = memory.need_bytes(CLIP_720P_8S, 518)
    monkeypatch.setenv(memory.OVERRIDE_ENV, str(need / GB))
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.refusal is None
    # One byte of rounding is the float round trip through the setting, not a real shortfall.
    assert plan.gpu_limit >= memory._pool_need(720, 1280, 518) - 1


def test_a_mac_that_will_not_say_does_not_start_a_pass(monkeypatch):
    """`vm_stat` timing out is what a stalled Mac looks like, the worst moment to add to it."""
    monkeypatch.setattr(memory, "_MAC", True)
    _machine(monkeypatch, 24, None)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.allowed == 0
    assert plan.refusal is not None and "would not report" in plan.refusal
    assert memory.OVERRIDE_ENV in plan.refusal
    monkeypatch.setenv(memory.OVERRIDE_ENV, "9")
    assert memory.plan(CLIP_720P_8S, 518).refusal is None


def test_other_systems_have_no_vm_stat_so_get_half_their_memory(monkeypatch):
    monkeypatch.setattr(memory, "_MAC", False)
    _machine(monkeypatch, 24, None)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.allowed == 12 * GB and plan.refusal is None

    _machine(monkeypatch, 8, None)
    refusal = memory.plan(CLIP_720P_8S, 518).refusal
    assert refusal is not None and "would not report" in refusal


def test_override_runs_what_the_machine_would_refuse(monkeypatch):
    _machine(monkeypatch, 24, 3)
    assert memory.plan(CLIP_720P_8S, 518).refusal is not None
    monkeypatch.setenv(memory.OVERRIDE_ENV, "9")
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.refusal is None and plan.allowed == 9 * GB


def test_override_too_low_still_refuses_and_says_who_set_it(monkeypatch):
    _machine(monkeypatch, 64, 40)
    monkeypatch.setenv(memory.OVERRIDE_ENV, "5.5")
    refusal = memory.plan(CLIP_720P_8S, 518).refusal
    assert refusal is not None and f"{memory.OVERRIDE_ENV} allows 5.5 GB" in refusal


@pytest.mark.parametrize("bad", ["", "  ", "lots", "0", "-4", "nan", "inf"])
def test_override_that_makes_no_sense_is_ignored(monkeypatch, bad):
    _machine(monkeypatch, 24, 3)
    monkeypatch.setenv(memory.OVERRIDE_ENV, bad)
    plan = memory.plan(CLIP_720P_8S, 518)
    assert plan.refusal is not None and "3.0 GB free" in plan.refusal


def test_pool_follows_the_picture_size_and_upstreams_wide_clip_rule():
    default = memory._pool_need(720, 1280, 518)
    assert memory._pool_need(720, 1280, 252) < default / 2
    # Portrait and landscape go through the model at the same size.
    assert memory._pool_need(1280, 720, 518) == default
    # Past 1.78:1 upstream shrinks the input so the long side stays bounded.
    assert memory._pool_need(720, 2560, 518) < memory._pool_need(720, 1280, 518) * 1.1


def test_longer_clips_need_more_because_their_buffers_are_bigger():
    assert memory.need_bytes((960, 720, 1280, 3), 518) > memory.need_bytes(CLIP_720P_8S, 518)
