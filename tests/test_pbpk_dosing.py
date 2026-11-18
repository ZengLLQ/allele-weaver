"""Tests for RKF45 PBPK/PD ODE integrator, mass-balance conservation, and dose optimization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from allele_weaver.catalog import get_drug_pk_params
from allele_weaver.models import DosingRegimen, NumericalInstabilityError
from allele_weaver.pbpk_dosing import (
    integrate_rkf45,
    optimize_precision_dose,
    simulate_pbpk_regimen,
)


def test_rkf45_analytical_exponential_convergence() -> None:
    # Linear 2-compartment cascade: dy0/dt = -k*y0, dy1/dt = k*y0 - k*y1
    k = 0.75
    y0 = np.array([100.0, 0.0], dtype=float)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.array([-k * y[0], k * y[0] - k * y[1]], dtype=float)

    ts, ys, acc, _ = integrate_rkf45(rhs, (0.0, 4.0), y0, rtol=1e-8, atol=1e-10)
    assert acc > 5
    t_end = ts[-1]
    exact_0 = 100.0 * math.exp(-k * t_end)
    exact_1 = 100.0 * k * t_end * math.exp(-k * t_end)
    assert ys[-1, 0] == pytest.approx(exact_0, rel=1e-6)
    assert ys[-1, 1] == pytest.approx(exact_1, rel=1e-6)


def test_pbpk_mass_balance_conservation_across_all_drugs() -> None:
    for drug_name, dose_mg in [
        ("warfarin", 5.0),
        ("codeine", 30.0),
        ("clopidogrel", 75.0),
        ("fluorouracil", 250.0),
    ]:
        drug = get_drug_pk_params(drug_name)
        reg = DosingRegimen(dose_mg=dose_mg, interval_h=24.0, n_doses=4)
        for as_val in (0.5, 1.0, 2.0):
            sim = simulate_pbpk_regimen(drug, reg, activity_score=as_val)
            assert sim.max_mass_balance_error_mg < 1e-3
            assert all(c >= -1e-9 for c in sim.c_plasma_mg_per_l)
            assert all(0.0 <= e <= 1.0 for e in sim.pd_effect)


def test_parent_drug_vs_prodrug_activity_score_monotonicity() -> None:
    # Parent drug cleared by CYP2C9 (warfarin): lower AS -> HIGHER AUC
    warfarin = get_drug_pk_params("warfarin")
    reg_w = DosingRegimen(dose_mg=5.0, interval_h=24.0, n_doses=8)
    sim_pm = simulate_pbpk_regimen(warfarin, reg_w, activity_score=0.5, diplotype_label="*3/*3")
    sim_nm = simulate_pbpk_regimen(warfarin, reg_w, activity_score=2.0, diplotype_label="*1/*1")
    assert sim_pm.auc_last_interval > sim_nm.auc_last_interval * 1.4
    assert sim_pm.c_max > sim_nm.c_max

    # Prodrug activated by CYP2D6 (codeine -> morphine): higher AS -> HIGHER active metabolite AUC
    codeine = get_drug_pk_params("codeine")
    reg_c = DosingRegimen(dose_mg=30.0, interval_h=8.0, n_doses=6)
    sim_im = simulate_pbpk_regimen(codeine, reg_c, activity_score=0.5, diplotype_label="*4/*41")
    sim_um = simulate_pbpk_regimen(codeine, reg_c, activity_score=3.0, diplotype_label="*1/*1x2")
    assert sim_um.auc_last_interval > sim_im.auc_last_interval * 2.0


def test_constrained_precision_dose_optimization_reduces_warfarin_dose_for_im() -> None:
    warfarin = get_drug_pk_params("warfarin")
    opt_nm = optimize_precision_dose(warfarin, activity_score=2.0, diplotype_label="*1/*1", nominal_dose_mg=5.0)
    opt_im = optimize_precision_dose(warfarin, activity_score=1.0, diplotype_label="*1/*3", nominal_dose_mg=5.0)

    assert opt_nm.feasible is True
    assert opt_im.feasible is True
    assert opt_im.optimal_dose_mg < opt_nm.optimal_dose_mg
    assert opt_im.predicted_c_max <= warfarin.c_tox_max_mg_per_l
    assert opt_im.predicted_ttr >= 0.70


def test_prodrug_poor_metabolizer_flags_infeasible() -> None:
    codeine = get_drug_pk_params("codeine")
    opt_pm = optimize_precision_dose(codeine, activity_score=0.0, diplotype_label="*4/*4", nominal_dose_mg=30.0)
    assert opt_pm.feasible is False
    assert opt_pm.optimal_dose_mg == 0.0

    with pytest.raises(NumericalInstabilityError):
        integrate_rkf45(lambda t, y: y, (5.0, 2.0), np.array([1.0]))
