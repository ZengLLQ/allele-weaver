"""Tests for Bayesian combinatorial star-allele diplotype inference & CPIC phenotyping."""

from __future__ import annotations

import pytest

from allele_weaver.cnv_hmm import decode_cnv_profile
from allele_weaver.models import MetabolizerPhenotype
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.star_caller import call_star_alleles, classify_metabolizer_phenotype


def test_diplotype_posteriors_sum_to_one_across_full_space(cyp2c19_locus, generator) -> None:
    sim_locus, frags, bins, h1, h2 = generator.simulate_sample(
        cyp2c19_locus, "*1", "*2", mean_depth=45.0
    )
    phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2))
    cnv = decode_cnv_profile("CYP2C19", bins)
    n_alleles = len(sim_locus.star_alleles)
    total_pairs = n_alleles * (n_alleles + 1) // 2

    all_calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv, top_k=total_pairs)
    assert len(all_calls) == total_pairs
    assert sum(c.posterior_prob for c in all_calls) == pytest.approx(1.0, abs=1e-6)
    assert all_calls[0].label == "*1/*2"
    assert all_calls[0].phenotype == MetabolizerPhenotype.INTERMEDIATE


def test_cyp2d6_star1_star4_vs_duplication_and_deletion(cyp2d6_locus, generator) -> None:
    for a1, a2, expected_pheno in [
        ("*1", "*4", MetabolizerPhenotype.INTERMEDIATE),
        ("*4", "*4", MetabolizerPhenotype.POOR),
        ("*1", "*5", MetabolizerPhenotype.INTERMEDIATE),
        ("*1", "*1x2", MetabolizerPhenotype.ULTRARAPID),
        ("*10", "*36", MetabolizerPhenotype.INTERMEDIATE),
    ]:
        sim_locus, frags, bins, h1, h2 = generator.simulate_sample(
            cyp2d6_locus, a1, a2, mean_depth=55.0, seq_error_rate=0.01
        )
        phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2))
        cnv = decode_cnv_profile("CYP2D6", bins)
        calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv, top_k=5)
        expected_label = "/".join(sorted((a1, a2)))
        assert calls[0].label == expected_label
        assert calls[0].phenotype == expected_pheno
        assert calls[0].posterior_prob >= 0.50


def test_cpic_phenotype_boundaries() -> None:
    assert classify_metabolizer_phenotype("CYP2D6", 0.0) == MetabolizerPhenotype.POOR
    assert classify_metabolizer_phenotype("CYP2D6", 1.0) == MetabolizerPhenotype.INTERMEDIATE
    assert classify_metabolizer_phenotype("CYP2D6", 2.0) == MetabolizerPhenotype.NORMAL
    assert classify_metabolizer_phenotype("CYP2D6", 3.0) == MetabolizerPhenotype.ULTRARAPID

    assert classify_metabolizer_phenotype("CYP2C19", 2.5) == MetabolizerPhenotype.RAPID
    assert classify_metabolizer_phenotype("CYP2C19", 3.0) == MetabolizerPhenotype.ULTRARAPID
    assert classify_metabolizer_phenotype("DPYD", 0.5) == MetabolizerPhenotype.POOR
    assert classify_metabolizer_phenotype("DPYD", 1.5) == MetabolizerPhenotype.INTERMEDIATE
