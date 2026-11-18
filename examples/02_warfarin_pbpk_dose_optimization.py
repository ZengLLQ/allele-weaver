"""Example 02: Genotype-guided warfarin PBPK/PD simulation and constrained dose optimization."""

from __future__ import annotations

from allele_weaver.catalog import get_cpic_guideline, get_drug_pk_params
from allele_weaver.models import DosingRegimen
from allele_weaver.pbpk_dosing import optimize_precision_dose, simulate_pbpk_regimen
from allele_weaver.star_caller import classify_metabolizer_phenotype
from allele_weaver.visualizer import render_pk_curve


def main() -> None:
    warfarin = get_drug_pk_params("warfarin")
    empiric_regimen = DosingRegimen(dose_mg=5.0, interval_h=24.0, n_doses=10)

    scenarios = [
        ("*1/*1", 2.0),
        ("*1/*2", 1.5),
        ("*1/*3", 1.0),
        ("*3/*3", 0.5),
    ]

    for dip_label, act_score in scenarios:
        pheno = classify_metabolizer_phenotype("CYP2C9", act_score)
        cpic = get_cpic_guideline("CYP2C9", "warfarin", pheno)
        empiric_sim = simulate_pbpk_regimen(
            warfarin, empiric_regimen, activity_score=act_score, diplotype_label=dip_label
        )
        opt = optimize_precision_dose(
            warfarin, activity_score=act_score, diplotype_label=dip_label, nominal_dose_mg=5.0
        )

        print(f"\n>>> {dip_label} | AS={act_score:.2f} | {pheno.value}")
        print(f"CPIC ({cpic.therapeutic_alert_level}): {cpic.recommendation}")
        print(
            f"Empiric 5.0 mg/day -> C_max={empiric_sim.c_max:.3f} mg/L, "
            f"AUC={empiric_sim.auc_last_interval:.2f}, TTR={empiric_sim.time_in_therapeutic_window_frac * 100:.1f}%"
        )
        print(
            f"Optimized {opt.optimal_dose_mg:.2f} mg q{opt.optimal_interval_h:.0f}h -> "
            f"C_max={opt.predicted_c_max:.3f} mg/L, TTR={opt.predicted_ttr * 100:.1f}%, "
            f"MassErr={opt.simulation.max_mass_balance_error_mg:.2e} mg"
        )
        print(render_pk_curve(opt.simulation, warfarin))


if __name__ == "__main__":
    main()
