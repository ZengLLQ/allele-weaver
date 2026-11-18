"""Tests for population pharmacogenetics (HWE, LD, EM haplotypes) and UQ."""

from __future__ import annotations

import pytest

from allele_weaver.catalog import get_drug_pk_params
from allele_weaver.models import (
    DiplotypeCall,
    DosingRegimen,
    MetabolizerPhenotype,
    NumericalInstabilityError,
)
from allele_weaver.popgen_uq import (
    analyze_pk_sensitivity,
    analyze_population_ld,
    em_haplotype_frequencies,
    hwe_exact_test,
    pairwise_ld,
    propagate_diplotype_to_pk_uq,
)


def test_hwe_exact_test_calibration() -> None:
    # Perfect HWE: p=0.5, n=100 -> 25, 50, 25
    assert hwe_exact_test(25, 50, 25) > 0.90
    # Extreme heterozygote deficit -> strong HWE violation
    assert hwe_exact_test(50, 0, 50) < 1e-10
    with pytest.raises(NumericalInstabilityError):
        hwe_exact_test(-1, 5, 5)


def test_pairwise_ld_perfect_and_independent() -> None:
    g1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    dp, r2 = pairwise_ld(g1, g1)
    assert dp == pytest.approx(1.0, abs=1e-4)
    assert r2 == pytest.approx(1.0, abs=1e-4)

    # Constant monomorphic site yields 0 LD
    g_const = [0] * len(g1)
    dp0, r20 = pairwise_ld(g1, g_const)
    assert dp0 == 0.0 and r20 == 0.0


def test_em_haplotype_frequencies_and_population_panel(cyp2d6_locus, generator) -> None:
    panel = generator.simulate_population_panel(cyp2d6_locus, n_individuals=180)
    haps, ll, iters = em_haplotype_frequencies(panel)
    assert iters >= 1
    assert ll < 0.0
    assert sum(f for _, f in haps) == pytest.approx(1.0, abs=0.02)

    ld = analyze_population_ld(panel)
    assert len(ld.hwe_p_values) == len(cyp2d6_locus.variants)
    for i in range(len(ld.rsids)):
        assert ld.r_squared[i][i] == pytest.approx(1.0, abs=1e-6)
        assert ld.d_prime[i][i] == pytest.approx(1.0, abs=1e-6)
        for j in range(len(ld.rsids)):
            assert 0.0 <= ld.r_squared[i][j] <= 1.0 + 1e-9
            assert 0.0 <= ld.d_prime[i][j] <= 1.0 + 1e-9


def test_monte_carlo_uq_and_sensitivity_ranking() -> None:
    warfarin = get_drug_pk_params("warfarin")
    reg = DosingRegimen(dose_mg=4.0, interval_h=24.0, n_doses=6)
    calls = [
        DiplotypeCall("*1/*3", "*3", 0.85, -10.0, -2.0, 1.0, MetabolizerPhenotype.INTERMEDIATE, True, True),
        DiplotypeCall("*1/*1", "*1", 0.15, -12.0, -1.0, 2.0, MetabolizerPhenotype.NORMAL, True, True),
    ]
    uq = propagate_diplotype_to_pk_uq(warfarin, reg, calls, n_draws=40, iiv_cv=0.15, seed=7)
    assert uq.auc_p05 <= uq.auc_p50 <= uq.auc_p95
    assert 0.0 <= uq.prob_toxicity <= 1.0
    assert 0.0 <= uq.prob_subtherapeutic <= 1.0

    sens = analyze_pk_sensitivity(warfarin, reg, activity_score=1.0)
    # Increasing bioavailability increases AUC (+1.0), increasing Vmax decreases AUC (< 0)
    assert sens["bioavailability"] > 0.8
    assert sens["vmax_cyp_mg_per_h"] < -0.5

    with pytest.raises(NumericalInstabilityError):
        propagate_diplotype_to_pk_uq(warfarin, reg, [], n_draws=10)
