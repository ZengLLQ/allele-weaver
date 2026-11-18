"""Tests for read-backed Minimum Error Correction (MEC) haplotype phasing."""

from __future__ import annotations

from typing import List

import pytest

from allele_weaver.models import PhasingConvergenceError, ReadFragment
from allele_weaver.phasing import (
    compute_block_n50,
    compute_switch_error,
    extract_phase_components,
    phase_locus_reads,
)


def _make_clean_reads(h0: List[int], h1: List[int], copies_per_hap: int = 3) -> List[ReadFragment]:
    frags: List[ReadFragment] = []
    n = len(h0)
    rid = 0
    for hap_idx, hap in enumerate((h0, h1)):
        for _ in range(copies_per_hap):
            for start in range(0, n - 1):
                end = min(n, start + 3)
                obs = {i: (hap[i], 35) for i in range(start, end)}
                frags.append(ReadFragment(f"r_{hap_idx}_{rid}", obs, mapping_quality=60))
                rid += 1
    return frags


def test_exact_bitmask_dp_recovers_zero_error_haplotypes(cyp2d6_locus) -> None:
    # Subset to first 5 sites with low spanning depth <= 10 so exact DP triggers
    h0 = [0, 1, 0, 1, 0, 0, 0, 0, 1]
    h1 = [1, 0, 1, 0, 0, 0, 1, 0, 0]
    frags: List[ReadFragment] = []
    # Two clean reads per haplotype across overlapping windows (max active <= 6)
    for k in range(2):
        frags.append(ReadFragment(f"a0_{k}", {0: (h0[0], 38), 1: (h0[1], 38), 2: (h0[2], 38)}))
        frags.append(ReadFragment(f"a1_{k}", {0: (h1[0], 38), 1: (h1[1], 38), 2: (h1[2], 38)}))
        frags.append(ReadFragment(f"b0_{k}", {2: (h0[2], 38), 3: (h0[3], 38), 6: (h0[6], 38)}))
        frags.append(ReadFragment(f"b1_{k}", {2: (h1[2], 38), 3: (h1[3], 38), 6: (h1[6], 38)}))
        frags.append(ReadFragment(f"c0_{k}", {6: (h0[6], 38), 8: (h0[8], 38)}))
        frags.append(ReadFragment(f"c1_{k}", {6: (h1[6], 38), 8: (h1[8], 38)}))

    res = phase_locus_reads(cyp2d6_locus, frags, exact_dp_max_reads=12, truth_haplotypes=(h0, h1))
    assert "exact_bitmask_dp" in res.solver_method
    assert res.total_unweighted_flips == 0
    assert res.switch_error_rate == 0.0
    assert res.hamming_error_rate == 0.0


def test_exact_dp_flips_single_low_phred_sequencing_error(cyp2c19_locus) -> None:
    h0 = [0, 1, 0, 1, 0]
    h1 = [1, 0, 1, 0, 1]
    frags = [
        ReadFragment("h0_1", {0: (0, 38), 1: (1, 38), 2: (0, 38)}),
        ReadFragment("h0_2", {0: (0, 38), 1: (1, 38), 2: (0, 38)}),
        ReadFragment("h0_3", {1: (1, 38), 2: (0, 38), 3: (1, 38), 4: (0, 38)}),
        # One read carries a single low-quality base error at site 2
        ReadFragment("h0_err", {1: (1, 36), 2: (1, 8), 3: (1, 36)}),
        ReadFragment("h1_1", {0: (1, 38), 1: (0, 38), 2: (1, 38)}),
        ReadFragment("h1_2", {0: (1, 38), 1: (0, 38), 2: (1, 38)}),
        ReadFragment("h1_3", {1: (0, 38), 2: (1, 38), 3: (0, 38), 4: (1, 38)}),
    ]
    res = phase_locus_reads(cyp2c19_locus, frags, exact_dp_max_reads=12, truth_haplotypes=(h0, h1))
    assert res.total_unweighted_flips == 1
    assert res.switch_error_rate == 0.0


def test_annealed_solver_handles_high_coverage_and_paralog_noise(cyp2d6_locus, generator) -> None:
    sim_locus, frags, _, h1, h2 = generator.simulate_sample(
        cyp2d6_locus, "*1", "*4", mean_depth=55.0, seq_error_rate=0.015
    )
    res = phase_locus_reads(sim_locus, frags, exact_dp_max_reads=10, truth_haplotypes=(h1, h2))
    assert res.phased_fraction >= 0.85
    assert res.n50_bp > 1000
    assert res.switch_error_rate is not None and res.switch_error_rate <= 0.10


def test_disconnected_components_and_n50_metrics() -> None:
    frags = [
        ReadFragment("c1", {0: (0, 30), 1: (1, 30)}),
        ReadFragment("c2", {3: (1, 30), 4: (0, 30)}),
        ReadFragment("singleton", {2: (0, 30)}),
    ]
    comps = extract_phase_components(6, frags)
    assert comps == [[0, 1], [3, 4]]
    assert compute_block_n50([1000, 4000, 500]) == 4000
    assert compute_block_n50([]) == 0


def test_switch_error_swap_invariance_and_errors() -> None:
    t0 = [0, 1, 0, 1]
    t1 = [1, 0, 1, 0]
    # Swapped global assignment must have 0 switch and 0 Hamming error
    sw, hm = compute_switch_error(t1, t0, t0, t1)
    assert sw == 0.0 and hm == 0.0
    # Single switch at midpoint
    sw2, _ = compute_switch_error([0, 1, 1, 0], [1, 0, 0, 1], t0, t1)
    assert sw2 == pytest.approx(1.0 / 3.0)

    with pytest.raises(PhasingConvergenceError):
        compute_switch_error([0, 1], [1, 0], t0, t1)
