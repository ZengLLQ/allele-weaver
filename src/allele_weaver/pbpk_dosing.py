"""Pharmacogene-scaled multi-compartment PBPK/PD solver & precision dose optimizer.

Implements a 6-state physiologically based pharmacokinetic / indirect-response
pharmacodynamic (PBPK/PD) ODE system:
    State y = [A_gut (mg), A_hep (mg), A_central (mg), A_periph (mg),
               A_met (mg), E_pd (dimensionless)]

Coupled equations:
    dA_gut/dt     = -k_a * A_gut
    dA_hep/dt     = k_a * F * A_gut + Q_h * (C_p - C_h) - V_max_eff * C_h / (K_m + C_h)
    dA_central/dt = Q_h * (C_h - C_p) - Q_p * (C_p - C_per) - CL_renal * C_p
    dA_periph/dt  = Q_p * (C_p - C_per)
    dA_met/dt     = f_m * [V_max_eff * C_h / (K_m + C_h)] - CL_met * C_met
    dE_pd/dt      = k_out * (E_max * C_eff^n / (EC50^n + C_eff^n) - E_pd)

where:
    V_max_eff = V_max_ref * (AS / AS_ref)^gamma

Integrated using a custom adaptive Runge-Kutta-Fehlberg 4(5) (RKF45) ODE solver
with error-tolerance step-size control, bolus dosing discontinuities, and exact
cumulative mass-balance conservation tracking!
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from allele_weaver.models import (
    DoseOptimizationResult,
    DosingRegimen,
    DrugPKParams,
    NumericalInstabilityError,
    PKSimulationResult,
)

# Standard Cash-Karp / Fehlberg 4(5) Butcher tableau coefficients
_RKF_A2 = 1.0 / 4.0
_RKF_A3 = 3.0 / 8.0
_RKF_A4 = 12.0 / 13.0
_RKF_A5 = 1.0
_RKF_A6 = 1.0 / 2.0

_RKF_B21 = 1.0 / 4.0
_RKF_B31, _RKF_B32 = 3.0 / 32.0, 9.0 / 32.0
_RKF_B41, _RKF_B42, _RKF_B43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
_RKF_B51, _RKF_B52, _RKF_B53, _RKF_B54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
_RKF_B61, _RKF_B62, _RKF_B63, _RKF_B64, _RKF_B65 = (
    -8.0 / 27.0,
    2.0,
    -3544.0 / 2565.0,
    1859.0 / 4104.0,
    -11.0 / 40.0,
)

# 5th-order weights
_RKF_C1, _RKF_C3, _RKF_C4, _RKF_C5, _RKF_C6 = (
    16.0 / 135.0,
    6656.0 / 12825.0,
    28561.0 / 56430.0,
    -9.0 / 50.0,
    2.0 / 55.0,
)
# Error estimate weights (5th - 4th order)
_RKF_E1, _RKF_E3, _RKF_E4, _RKF_E5, _RKF_E6 = (
    1.0 / 360.0,
    -128.0 / 4275.0,
    -2197.0 / 75240.0,
    1.0 / 50.0,
    2.0 / 55.0,
)


def integrate_rkf45(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    h_init: float = 0.05,
    h_min: float = 1e-5,
    h_max: float = 0.5,
    max_steps: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Integrate an ODE system dy/dt = rhs(t, y) via adaptive RKF45.

    Returns:
        (times, states, steps_accepted, steps_rejected)
    """
    t0, t_end = t_span
    if t_end <= t0:
        raise NumericalInstabilityError(f"Invalid t_span ({t0}, {t_end})")

    t = float(t0)
    y = np.array(y0, dtype=float, copy=True)
    h = min(h_init, t_end - t0)

    ts: List[float] = [t]
    ys: List[np.ndarray] = [y.copy()]
    accepted = 0
    rejected = 0

    for _ in range(max_steps):
        if t >= t_end - 1e-12:
            break
        if t + h > t_end:
            h = t_end - t

        k1 = rhs(t, y)
        k2 = rhs(t + _RKF_A2 * h, y + h * (_RKF_B21 * k1))
        k3 = rhs(t + _RKF_A3 * h, y + h * (_RKF_B31 * k1 + _RKF_B32 * k2))
        k4 = rhs(t + _RKF_A4 * h, y + h * (_RKF_B41 * k1 + _RKF_B42 * k2 + _RKF_B43 * k3))
        k5 = rhs(
            t + _RKF_A5 * h,
            y + h * (_RKF_B51 * k1 + _RKF_B52 * k2 + _RKF_B53 * k3 + _RKF_B54 * k4),
        )
        k6 = rhs(
            t + _RKF_A6 * h,
            y + h * (_RKF_B61 * k1 + _RKF_B62 * k2 + _RKF_B63 * k3 + _RKF_B64 * k4 + _RKF_B65 * k5),
        )

        y5 = y + h * (_RKF_C1 * k1 + _RKF_C3 * k3 + _RKF_C4 * k4 + _RKF_C5 * k5 + _RKF_C6 * k6)
        err_vec = h * (_RKF_E1 * k1 + _RKF_E3 * k3 + _RKF_E4 * k4 + _RKF_E5 * k5 + _RKF_E6 * k6)

        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

        if not np.isfinite(err_norm):
            raise NumericalInstabilityError("RKF45 encountered non-finite derivative or state")

        if err_norm <= 1.0 or h <= h_min * 1.01:
            t += h
            y = y5
            accepted += 1
            ts.append(t)
            ys.append(y.copy())

        if err_norm == 0.0:
            factor = 2.0
        else:
            factor = 0.84 * (1.0 / err_norm) ** 0.25

        if err_norm > 1.0 and h > h_min * 1.01:
            rejected += 1
            h = max(h_min, h * max(0.2, min(0.8, factor)))
        else:
            h = min(h_max, max(h_min, h * min(2.5, max(0.5, factor))))

    return np.asarray(ts, dtype=float), np.vstack(ys), accepted, rejected


def effective_vmax(
    drug: DrugPKParams,
    activity_score: float,
    reference_activity_score: float = 2.0,
) -> float:
    """Scale maximal CYP metabolic velocity nonlinearly by Activity Score."""
    if activity_score < 0.0 or reference_activity_score <= 0.0:
        raise NumericalInstabilityError("Invalid activity score for PBPK scaling")
    ratio = activity_score / reference_activity_score
    return float(drug.vmax_cyp_mg_per_h * (ratio ** drug.activity_exponent))


def simulate_pbpk_regimen(
    drug: DrugPKParams,
    regimen: DosingRegimen,
    activity_score: float,
    diplotype_label: str = "*1/*1",
    reference_activity_score: float = 2.0,
    inter_individual_cl_scale: float = 1.0,
) -> PKSimulationResult:
    """Simulate multi-dose PBPK/PD profile using RKF45 with mass-balance verification.

    Tracks cumulative drug absorbed, CYP-metabolized, and renally eliminated
    alongside compartment amounts so exact mass-balance error is audited on
    every run.
    """
    vmax_eff = effective_vmax(drug, activity_score, reference_activity_score) * inter_individual_cl_scale
    cl_renal_eff = drug.cl_renal_l_per_h * inter_individual_cl_scale

    # State vector (8 states: 6 PBPK/PD + 2 cumulative elimination trackers for mass balance):
    # [A_gut, A_hep, A_central, A_periph, A_met, E_pd, A_elim_cyp, A_elim_renal]
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        a_gut, a_hep, a_cen, a_per, a_met, e_pd, _, _ = np.maximum(0.0, state)
        c_hep = a_hep / drug.v_hepatic_l
        c_cen = a_cen / drug.v_central_l
        c_per = a_per / drug.v_periph_l
        c_met = a_met / drug.v_met_l

        rate_abs = drug.ka_per_h * a_gut
        # First-pass loss (1 - F) exits before portal entry
        rate_portal = drug.bioavailability * rate_abs
        rate_cyp = (vmax_eff * c_hep) / (drug.km_mg_per_l + c_hep)
        rate_renal = cl_renal_eff * c_cen
        flux_hep_to_cen = drug.q_hepatic_l_per_h * (c_hep - c_cen)
        flux_cen_to_per = drug.q_inter_l_per_h * (c_cen - c_per)

        d_gut = -rate_abs
        d_hep = rate_portal - flux_hep_to_cen - rate_cyp
        d_cen = flux_hep_to_cen - flux_cen_to_per - rate_renal
        d_per = flux_cen_to_per
        d_met = drug.fm_active_metabolite * rate_cyp - drug.cl_met_l_per_h * c_met

        c_eff = c_met if drug.is_prodrug else c_cen
        hill_num = c_eff ** drug.hill_n
        hill_den = (drug.ec50_mg_per_l ** drug.hill_n) + hill_num
        target_pd = drug.emax * (hill_num / hill_den) if hill_den > 0 else 0.0
        d_epd = drug.k_out_per_h * (target_pd - e_pd)

        return np.array(
            [d_gut, d_hep, d_cen, d_per, d_met, d_epd, rate_cyp, rate_renal],
            dtype=float,
        )

    y_cur = np.zeros(8, dtype=float)
    all_t: List[float] = []
    all_y: List[np.ndarray] = []
    tot_acc = 0
    tot_rej = 0
    total_bioavailable_input = 0.0

    for d_idx in range(regimen.n_doses):
        dose = regimen.loading_dose_mg if (d_idx == 0 and regimen.loading_dose_mg) else regimen.dose_mg
        y_cur[0] += dose
        total_bioavailable_input += dose * drug.bioavailability

        t0 = d_idx * regimen.interval_h
        t1 = (d_idx + 1) * regimen.interval_h
        ts, ys, acc, rej = integrate_rkf45(rhs, (t0, t1), y_cur, h_max=min(0.4, regimen.interval_h / 10.0))
        tot_acc += acc
        tot_rej += rej

        if d_idx == 0:
            all_t.extend(ts.tolist())
            all_y.extend([row for row in ys])
        else:
            all_t.extend(ts[1:].tolist())
            all_y.extend([row for row in ys[1:]])
        y_cur = ys[-1].copy()

    t_arr = np.asarray(all_t, dtype=float)
    y_mat = np.vstack(all_y)

    c_hep_arr = np.maximum(0.0, y_mat[:, 1]) / drug.v_hepatic_l
    c_cen_arr = np.maximum(0.0, y_mat[:, 2]) / drug.v_central_l
    c_per_arr = np.maximum(0.0, y_mat[:, 3]) / drug.v_periph_l
    c_met_arr = np.maximum(0.0, y_mat[:, 4]) / drug.v_met_l
    pd_arr = np.clip(y_mat[:, 5], 0.0, 1.0)

    # Mass-balance check at end of simulation:
    # Bioavailable mass entered = F * (sum(doses) - A_gut_remaining)
    bioavail_entered = total_bioavailable_input - drug.bioavailability * max(0.0, float(y_mat[-1, 0]))
    accounted_parent = float(
        max(0.0, y_mat[-1, 1])
        + max(0.0, y_mat[-1, 2])
        + max(0.0, y_mat[-1, 3])
        + max(0.0, y_mat[-1, 6])
        + max(0.0, y_mat[-1, 7])
    )
    mass_err = abs(bioavail_entered - accounted_parent)

    # Exposure metrics on active moiety during final dosing interval
    c_active = c_met_arr if drug.is_prodrug else c_cen_arr
    last_start = (regimen.n_doses - 1) * regimen.interval_h
    mask_last = t_arr >= last_start - 1e-9
    t_last = t_arr[mask_last]
    c_last = c_active[mask_last]

    c_max = float(np.max(c_last))
    c_trough = float(c_last[-1])
    trap_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_last = float(trap_fn(c_last, t_last))

    # Time in therapeutic window [c_eff_min, c_tox_max] via trapezoidal dwell fraction
    in_win = ((c_last >= drug.c_eff_min_mg_per_l) & (c_last <= drug.c_tox_max_mg_per_l)).astype(float)
    dt = np.diff(t_last)
    if len(dt) > 0:
        ttr = float(np.sum(0.5 * (in_win[:-1] + in_win[1:]) * dt) / max(1e-9, t_last[-1] - t_last[0]))
    else:
        ttr = float(in_win[0])

    return PKSimulationResult(
        drug_name=drug.drug_name,
        diplotype_label=diplotype_label,
        activity_score=activity_score,
        time_h=t_arr.tolist(),
        c_plasma_mg_per_l=c_cen_arr.tolist(),
        c_hepatic_mg_per_l=c_hep_arr.tolist(),
        c_periph_mg_per_l=c_per_arr.tolist(),
        c_metabolite_mg_per_l=c_met_arr.tolist(),
        pd_effect=pd_arr.tolist(),
        c_max=c_max,
        c_trough=c_trough,
        auc_last_interval=auc_last,
        time_in_therapeutic_window_frac=ttr,
        max_mass_balance_error_mg=mass_err,
        rkf_steps_taken=tot_acc,
        rkf_steps_rejected=tot_rej,
    )


def optimize_precision_dose(
    drug: DrugPKParams,
    activity_score: float,
    diplotype_label: str = "*1/*1",
    nominal_dose_mg: float = 10.0,
    candidate_intervals_h: Tuple[float, ...] = (12.0, 24.0),
    n_doses_to_ss: int = 8,
) -> DoseOptimizationResult:
    """Find optimal (dose_mg, interval_h) keeping peak/trough inside therapeutic window.

    Uses bounded Brent scalar minimization with quadratic penalty on toxicity
    ceiling violations (C_max > c_tox_max) and subtherapeutic troughs
    (C_trough < c_eff_min), targeting geometric midpoint exposure.
    """
    if activity_score <= 0.05 and drug.is_prodrug:
        # Prodrug cannot be bioactivated in Poor Metabolizers regardless of dose
        reg = DosingRegimen(dose_mg=nominal_dose_mg, interval_h=candidate_intervals_h[-1], n_doses=n_doses_to_ss)
        sim = simulate_pbpk_regimen(drug, reg, activity_score, diplotype_label=diplotype_label)
        return DoseOptimizationResult(
            drug_name=drug.drug_name,
            diplotype_label=diplotype_label,
            activity_score=activity_score,
            optimal_dose_mg=0.0,
            optimal_interval_h=candidate_intervals_h[-1],
            predicted_c_max=sim.c_max,
            predicted_c_trough=sim.c_trough,
            predicted_ttr=sim.time_in_therapeutic_window_frac,
            safety_margin_ratio=drug.c_tox_max_mg_per_l / max(1e-6, sim.c_max),
            feasible=False,
            simulation=sim,
        )

    c_mid = math.sqrt(drug.c_eff_min_mg_per_l * drug.c_tox_max_mg_per_l)
    best_obj = float("inf")
    best_dose = nominal_dose_mg
    best_interval = candidate_intervals_h[0]

    for interval_h in candidate_intervals_h:
        def objective(dose_mg: float) -> float:
            reg = DosingRegimen(dose_mg=float(dose_mg), interval_h=interval_h, n_doses=n_doses_to_ss)
            sim = simulate_pbpk_regimen(drug, reg, activity_score, diplotype_label=diplotype_label)
            c_avg = sim.auc_last_interval / interval_h
            loss = ((c_avg - c_mid) / c_mid) ** 2
            if sim.c_max > drug.c_tox_max_mg_per_l:
                loss += 50.0 * ((sim.c_max - drug.c_tox_max_mg_per_l) / drug.c_tox_max_mg_per_l) ** 2
            if sim.c_trough < drug.c_eff_min_mg_per_l:
                loss += 25.0 * ((drug.c_eff_min_mg_per_l - sim.c_trough) / drug.c_eff_min_mg_per_l) ** 2
            loss += 0.5 * (1.0 - sim.time_in_therapeutic_window_frac)
            return loss

        res = minimize_scalar(
            objective,
            bounds=(0.1 * nominal_dose_mg, 4.0 * nominal_dose_mg),
            method="bounded",
            options={"xatol": max(1e-3, 0.005 * nominal_dose_mg), "maxiter": 30},
        )
        if float(res.fun) < best_obj:
            best_obj = float(res.fun)
            best_dose = float(res.x)
            best_interval = interval_h

    # Round clinical tablet dose nicely to 0.25 mg increments when small, else 0.5 mg
    step = 0.25 if best_dose < 15.0 else 0.5
    clean_dose = max(step, round(best_dose / step) * step)
    final_reg = DosingRegimen(dose_mg=clean_dose, interval_h=best_interval, n_doses=n_doses_to_ss)
    final_sim = simulate_pbpk_regimen(drug, final_reg, activity_score, diplotype_label=diplotype_label)

    feasible = (
        final_sim.c_max <= drug.c_tox_max_mg_per_l
        and final_sim.c_trough >= 0.85 * drug.c_eff_min_mg_per_l
    )
    return DoseOptimizationResult(
        drug_name=drug.drug_name,
        diplotype_label=diplotype_label,
        activity_score=activity_score,
        optimal_dose_mg=clean_dose,
        optimal_interval_h=best_interval,
        predicted_c_max=final_sim.c_max,
        predicted_c_trough=final_sim.c_trough,
        predicted_ttr=final_sim.time_in_therapeutic_window_frac,
        safety_margin_ratio=drug.c_tox_max_mg_per_l / max(1e-6, final_sim.c_max),
        feasible=feasible,
        simulation=final_sim,
    )
