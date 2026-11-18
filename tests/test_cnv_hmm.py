"""Tests for structural CNV and paralog gene-conversion HMM decoding."""

from __future__ import annotations

import numpy as np
import pytest

from allele_weaver.cnv_hmm import (
    HMM_STATES,
    build_transition_matrix,
    calibrate_hmm_baseline,
    decode_cnv_profile,
)
from allele_weaver.models import CNVBinObservation, CNVState, NumericalInstabilityError


def test_forward_backward_posteriors_sum_to_one(cyp2d6_locus, generator) -> None:
    _, _, bins, _, _ = generator.simulate_sample(cyp2d6_locus, "*1", "*2", mean_depth=50.0)
    profile = decode_cnv_profile("CYP2D6", bins)

    post = np.asarray(profile.posterior_matrix, dtype=float)
    assert post.shape == (len(bins), len(HMM_STATES))
    np.testing.assert_allclose(post.sum(axis=1), np.ones(len(bins)), atol=1e-9)
    assert np.all(post >= 0.0) and np.all(post <= 1.0)
    assert np.isfinite(profile.log_likelihood)


def test_cnv_hmm_detects_heterozygous_deletion_and_duplication(cyp2d6_locus, generator) -> None:
    _, _, bins_del, _, _ = generator.simulate_sample(cyp2d6_locus, "*1", "*5", mean_depth=50.0)
    prof_del = decode_cnv_profile("CYP2D6", bins_del)
    assert prof_del.estimated_total_copies == pytest.approx(1.0, abs=0.4)
    assert any(s == CNVState.HET_DEL_1 for s in prof_del.viterbi_path)

    _, _, bins_dup, _, _ = generator.simulate_sample(cyp2d6_locus, "*1", "*1x2", mean_depth=50.0)
    prof_dup = decode_cnv_profile("CYP2D6", bins_dup)
    assert prof_dup.estimated_total_copies == pytest.approx(3.0, abs=0.45)
    assert any(s == CNVState.DUP_3 for s in prof_dup.viterbi_path)


def test_cnv_hmm_detects_homozygous_deletion_and_hybrid_conversion(cyp2d6_locus, generator) -> None:
    _, _, bins_null, _, _ = generator.simulate_sample(cyp2d6_locus, "*5", "*5", mean_depth=50.0)
    prof_null = decode_cnv_profile("CYP2D6", bins_null)
    assert prof_null.estimated_total_copies <= 0.35
    assert any(s == CNVState.DEL_0 for s in prof_null.viterbi_path)

    _, _, bins_hyb, _, _ = generator.simulate_sample(cyp2d6_locus, "*10", "*36", mean_depth=55.0)
    prof_hyb = decode_cnv_profile("CYP2D6", bins_hyb)
    assert prof_hyb.has_gene_conversion is True
    assert any(e.state == CNVState.HYBRID_CONV for e in prof_hyb.events)


def test_baseline_calibration_and_error_guards() -> None:
    bins = [
        CNVBinObservation(i, i * 1000, (i + 1) * 1000, observed_depth=55.0, expected_diploid_depth=50.0)
        for i in range(6)
    ]
    scale = calibrate_hmm_baseline(bins)
    assert scale == pytest.approx(1.10, abs=1e-6)

    trans = build_transition_matrix(stay_prob=0.92)
    np.testing.assert_allclose(trans.sum(axis=1), np.ones(len(HMM_STATES)), atol=1e-9)

    with pytest.raises(NumericalInstabilityError):
        decode_cnv_profile("EMPTY", [])
    with pytest.raises(NumericalInstabilityError):
        build_transition_matrix(stay_prob=0.3)
