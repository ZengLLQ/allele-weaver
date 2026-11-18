"""Additional clinical pharmacogenomics, numerical invariant, and multi-gene tests."""

from __future__ import annotations

import numpy as np
import pytest

from allele_weaver.catalog import (
    SyntheticCohortGenerator,
    get_cpic_guideline,
    get_drug_pk_params,
    get_pharmacogene_locus,
)
from allele_weaver.cnv_hmm import HMM_STATES, decode_cnv_profile
from allele_weaver.models import (
    CNVBinObservation,
    DosingRegimen,
    LocusDefinitionError,
    MetabolizerPhenotype,
    PharmacogeneLocus,
)
from allele_weaver.pbpk_dosing import integrate_rkf45, optimize_precision_dose, simulate_pbpk_regimen
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.star_caller import call_star_alleles


@pytest.mark.parametrize(
    "gene,a1,a2,expected_pheno",
    [
        ("CYP2D6", "*1", "*2", MetabolizerPhenotype.NORMAL),
        ("CYP2D6", "*4", "*41", MetabolizerPhenotype.INTERMEDIATE),
        ("CYP2C19", "*1", "*17", MetabolizerPhenotype.RAPID),
        ("CYP2C19", "*17", "*17", MetabolizerPhenotype.ULTRARAPID),
        ("CYP2C9", "*1", "*2", MetabolizerPhenotype.INTERMEDIATE),
        ("CYP2C9", "*3", "*3", MetabolizerPhenotype.POOR),
        ("DPYD", "*1", "*2A", MetabolizerPhenotype.INTERMEDIATE),
        ("DPYD", "*2A", "*13", MetabolizerPhenotype.POOR),
    ],
)
def test_multi_gene_diplotype_calling_and_cpic_phenotype(
    gene: str, a1: str, a2: str, expected_pheno: MetabolizerPhenotype
) -> None:
    locus = get_pharmacogene_locus(gene)
    gen = SyntheticCohortGenerator(seed=99)
    sim_locus, frags, bins, h1, h2 = gen.simulate_sample(
        locus, a1, a2, mean_depth=55.0, seq_error_rate=0.01
    )
    phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2), seed=99)
    cnv = decode_cnv_profile(gene, bins)
    calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv, top_k=3)

    expected_label = "/".join(sorted((a1, a2)))
    assert calls[0].label == expected_label
    assert calls[0].phenotype == expected_pheno


def test_cpic_guidelines_cover_critical_alerts() -> None:
    g_um = get_cpic_guideline("CYP2D6", "codeine", MetabolizerPhenotype.ULTRARAPID)
    assert g_um.therapeutic_alert_level == "CRITICAL_AVOID"
    assert g_um.dose_multiplier == 0.0

    g_clop = get_cpic_guideline("CYP2C19", "clopidogrel", MetabolizerPhenotype.POOR)
    assert g_clop.therapeutic_alert_level == "CRITICAL_AVOID"

    g_dpyd = get_cpic_guideline("DPYD", "fluorouracil", MetabolizerPhenotype.INTERMEDIATE)
    assert g_dpyd.dose_multiplier == pytest.approx(0.5)


def test_fluorouracil_dpyd_im_requires_dose_reduction() -> None:
    fu = get_drug_pk_params("fluorouracil")
    opt_nm = optimize_precision_dose(
        fu, activity_score=2.0, diplotype_label="*1/*1", nominal_dose_mg=150.0, candidate_intervals_h=(6.0, 12.0)
    )
    opt_im = optimize_precision_dose(
        fu, activity_score=1.0, diplotype_label="*1/*2A", nominal_dose_mg=150.0, candidate_intervals_h=(6.0, 12.0)
    )
    assert opt_nm.feasible is True
    assert opt_im.feasible is True
    daily_nm = opt_nm.optimal_dose_mg * (24.0 / opt_nm.optimal_interval_h)
    daily_im = opt_im.optimal_dose_mg * (24.0 / opt_im.optimal_interval_h)
    assert daily_im < daily_nm


def test_clopidogrel_prodrug_activation_scales_with_cyp2c19_as() -> None:
    clop = get_drug_pk_params("clopidogrel")
    reg = DosingRegimen(dose_mg=75.0, interval_h=24.0, n_doses=6)
    sim_pm = simulate_pbpk_regimen(clop, reg, activity_score=0.0, diplotype_label="*2/*2")
    sim_nm = simulate_pbpk_regimen(clop, reg, activity_score=2.0, diplotype_label="*1/*1")
    sim_um = simulate_pbpk_regimen(clop, reg, activity_score=3.0, diplotype_label="*17/*17")

    assert sim_pm.auc_last_interval == pytest.approx(0.0, abs=1e-6)
    assert sim_um.auc_last_interval > sim_nm.auc_last_interval


def test_rkf45_tolerance_refinement_improves_accuracy() -> None:
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.array([-0.6 * y[0]], dtype=float)

    y0 = np.array([10.0], dtype=float)
    exact = 10.0 * np.exp(-0.6 * 5.0)
    _, ys_coarse, _, _ = integrate_rkf45(rhs, (0.0, 5.0), y0, rtol=1e-3, atol=1e-5)
    _, ys_fine, _, _ = integrate_rkf45(rhs, (0.0, 5.0), y0, rtol=1e-8, atol=1e-10)
    err_coarse = abs(ys_coarse[-1, 0] - exact)
    err_fine = abs(ys_fine[-1, 0] - exact)
    assert err_fine < 1e-6
    assert err_fine <= err_coarse + 1e-12


def test_viterbi_and_marginal_posterior_agreement_on_neutral_profile() -> None:
    bins = [
        CNVBinObservation(i, i * 500, (i + 1) * 500, observed_depth=50.0, expected_diploid_depth=50.0)
        for i in range(8)
    ]
    prof = decode_cnv_profile("CYP2C9", bins, calibrate_baseline=True)
    post = np.asarray(prof.posterior_matrix)
    for t, v_state in enumerate(prof.viterbi_path):
        assert HMM_STATES[int(np.argmax(post[t]))] == v_state


def test_unknown_gene_or_drug_raises_locus_definition_error() -> None:
    with pytest.raises(LocusDefinitionError):
        get_pharmacogene_locus("UNKNOWN_GENE")
    with pytest.raises(LocusDefinitionError):
        get_drug_pk_params("aspirin")
    with pytest.raises(LocusDefinitionError):
        PharmacogeneLocus("TEST", "chr1", 200, 100, [], {})
