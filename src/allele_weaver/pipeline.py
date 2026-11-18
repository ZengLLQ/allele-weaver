"""End-to-end pharmacogenomics decision-support pipeline and cohort benchmarking.

Orchestrates read-backed MEC haplotype phasing, structural CNV/hybrid HMM
decoding, Bayesian star-allele diplotype calling, CPIC guideline matching,
genotype-guided RKF45 PBPK/PD dose optimization, and Monte Carlo UQ into
a unified ClinicalReport object with JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from allele_weaver.catalog import (
    SyntheticCohortGenerator,
    expected_cnv_state_for_diplotype,
    get_cpic_guideline,
    get_drug_pk_params,
    get_pharmacogene_locus,
)
from allele_weaver.cnv_hmm import decode_cnv_profile
from allele_weaver.models import (
    CNVBinObservation,
    CNVPosteriorProfile,
    CNVState,
    CPICGuideline,
    DiplotypeCall,
    DoseOptimizationResult,
    DosingRegimen,
    DrugPKParams,
    PharmacogeneLocus,
    PhasingResult,
    ReadFragment,
    UncertaintySummary,
)
from allele_weaver.pbpk_dosing import optimize_precision_dose
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.popgen_uq import propagate_diplotype_to_pk_uq
from allele_weaver.star_caller import call_star_alleles
from allele_weaver.visualizer import (
    render_cnv_profile,
    render_diplotype_posteriors,
    render_haplotype_tracks,
    render_pk_curve,
)


@dataclass
class ClinicalReport:
    """Comprehensive genomic-to-PBPK precision medicine evaluation."""

    gene_symbol: str
    drug_name: str
    phasing: PhasingResult
    cnv_profile: CNVPosteriorProfile
    diplotype_calls: List[DiplotypeCall]
    top_call: DiplotypeCall
    cpic_guideline: CPICGuideline
    dose_optimization: DoseOptimizationResult
    uncertainty: UncertaintySummary
    ascii_dashboard: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full clinical decision report into JSON-compatible dict."""
        return {
            "gene_symbol": self.gene_symbol,
            "drug_name": self.drug_name,
            "top_diplotype": self.top_call.label,
            "posterior_probability": round(self.top_call.posterior_prob, 6),
            "activity_score": self.top_call.activity_score,
            "phenotype": self.top_call.phenotype.value,
            "cpic_alert_level": self.cpic_guideline.therapeutic_alert_level,
            "cpic_recommendation": self.cpic_guideline.recommendation,
            "dose_multiplier": self.cpic_guideline.dose_multiplier,
            "phasing": {
                "solver_method": self.phasing.solver_method,
                "n_blocks": len(self.phasing.blocks),
                "phased_fraction": round(self.phasing.phased_fraction, 4),
                "n50_bp": self.phasing.n50_bp,
                "total_mec_cost": round(self.phasing.total_mec_cost, 4),
                "switch_error_rate": self.phasing.switch_error_rate,
            },
            "cnv": {
                "estimated_total_copies": round(self.cnv_profile.estimated_total_copies, 3),
                "has_gene_conversion": self.cnv_profile.has_gene_conversion,
                "log_likelihood": round(self.cnv_profile.log_likelihood, 3),
                "events": [
                    {
                        "state": e.state.value,
                        "bins": [e.start_bin, e.end_bin],
                        "mean_depth_ratio": round(e.mean_depth_ratio, 3),
                        "mean_posterior": round(e.mean_posterior_prob, 4),
                    }
                    for e in self.cnv_profile.events
                ],
            },
            "dose_optimization": {
                "optimal_dose_mg": self.dose_optimization.optimal_dose_mg,
                "optimal_interval_h": self.dose_optimization.optimal_interval_h,
                "predicted_c_max": round(self.dose_optimization.predicted_c_max, 5),
                "predicted_c_trough": round(self.dose_optimization.predicted_c_trough, 5),
                "predicted_ttr": round(self.dose_optimization.predicted_ttr, 4),
                "safety_margin_ratio": round(self.dose_optimization.safety_margin_ratio, 3),
                "feasible": self.dose_optimization.feasible,
                "mass_balance_error_mg": self.dose_optimization.simulation.max_mass_balance_error_mg,
            },
            "uncertainty": {
                "n_draws": self.uncertainty.n_draws,
                "auc_p05": round(self.uncertainty.auc_p05, 4),
                "auc_p50": round(self.uncertainty.auc_p50, 4),
                "auc_p95": round(self.uncertainty.auc_p95, 4),
                "prob_toxicity": round(self.uncertainty.prob_toxicity, 4),
                "prob_subtherapeutic": round(self.uncertainty.prob_subtherapeutic, 4),
                "sensitivity_indices": {
                    k: round(v, 4) for k, v in self.uncertainty.sensitivity_indices.items()
                },
            },
            "candidate_diplotypes": [
                {
                    "diplotype": c.label,
                    "posterior_prob": round(c.posterior_prob, 6),
                    "activity_score": c.activity_score,
                    "phenotype": c.phenotype.value,
                    "phase_consistent": c.phase_consistent,
                    "cnv_consistent": c.cnv_consistent,
                }
                for c in self.diplotype_calls
            ],
        }


class PharmacogenomicsPipeline:
    """High-level orchestrator coupling genomic calling to PBPK dose design."""

    def __init__(self, exact_dp_max_reads: int = 12, uq_draws: int = 60, seed: int = 42) -> None:
        self.exact_dp_max_reads = exact_dp_max_reads
        self.uq_draws = uq_draws
        self.seed = seed

    def run(
        self,
        locus: PharmacogeneLocus,
        fragments: Sequence[ReadFragment],
        cnv_bins: Sequence[CNVBinObservation],
        drug_name: str,
        nominal_dose_mg: float = 30.0,
        truth_haplotypes: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    ) -> ClinicalReport:
        """Execute full phasing -> CNV HMM -> star calling -> PBPK -> UQ workflow."""
        phasing = phase_locus_reads(
            locus,
            fragments,
            exact_dp_max_reads=self.exact_dp_max_reads,
            truth_haplotypes=truth_haplotypes,
            seed=self.seed,
        )
        cnv_profile = decode_cnv_profile(locus.gene_symbol, cnv_bins)
        calls = call_star_alleles(locus, phasing=phasing, cnv_profile=cnv_profile, top_k=6)
        top_call = calls[0]

        drug: DrugPKParams = get_drug_pk_params(drug_name)
        guideline = get_cpic_guideline(locus.gene_symbol, drug.drug_name, top_call.phenotype)
        opt = optimize_precision_dose(
            drug,
            activity_score=top_call.activity_score,
            diplotype_label=top_call.label,
            nominal_dose_mg=nominal_dose_mg,
        )

        reg_for_uq = DosingRegimen(
            dose_mg=max(0.5, opt.optimal_dose_mg if opt.optimal_dose_mg > 0 else nominal_dose_mg),
            interval_h=opt.optimal_interval_h,
            n_doses=6,
        )
        uq = propagate_diplotype_to_pk_uq(
            drug, reg_for_uq, calls, n_draws=self.uq_draws, seed=self.seed
        )

        sections = [
            render_haplotype_tracks(locus, phasing),
            "",
            render_cnv_profile(cnv_profile),
            "",
            render_diplotype_posteriors(calls),
            "",
            f"=== CPIC CLINICAL GUIDELINE ({guideline.therapeutic_alert_level}) ===",
            f"  Gene/Drug : {guideline.gene_symbol} / {guideline.drug_name.upper()}",
            f"  Phenotype : {guideline.phenotype.value}",
            f"  Guidance  : {guideline.recommendation}",
            f"  Opt Dose  : {opt.optimal_dose_mg:.2f} mg q{opt.optimal_interval_h:.0f}h "
            f"(Feasible={opt.feasible}, SafetyMargin={opt.safety_margin_ratio:.2f}x)",
            f"  UQ AUC90% : [{uq.auc_p05:.3f}, {uq.auc_p95:.3f}] mg*h/L | "
            f"P(Tox)={uq.prob_toxicity:.2f} | P(Sub)={uq.prob_subtherapeutic:.2f}",
            "",
            render_pk_curve(opt.simulation, drug),
        ]
        dashboard = "\n".join(sections)

        return ClinicalReport(
            gene_symbol=locus.gene_symbol,
            drug_name=drug.drug_name,
            phasing=phasing,
            cnv_profile=cnv_profile,
            diplotype_calls=calls,
            top_call=top_call,
            cpic_guideline=guideline,
            dose_optimization=opt,
            uncertainty=uq,
            ascii_dashboard=dashboard,
        )


def run_benchmark_suite(seed: int = 42, trials_per_case: int = 4) -> Dict[str, Any]:
    """Benchmark MEC phasing, CNV HMM recall, and diplotype accuracy on tricky clinical cases."""
    gen = SyntheticCohortGenerator(seed=seed)
    cases = [
        ("CYP2D6", "*1", "*4"),
        ("CYP2D6", "*1", "*10"),
        ("CYP2D6", "*4", "*41"),
        ("CYP2D6", "*1", "*5"),
        ("CYP2D6", "*1", "*1x2"),
        ("CYP2D6", "*10", "*36"),
        ("CYP2C19", "*1", "*2"),
        ("CYP2C19", "*2", "*17"),
        ("CYP2C9", "*1", "*3"),
        ("DPYD", "*1", "*2A"),
    ]

    total = 0
    diplotype_hits = 0
    cnv_hits = 0
    switch_errors: List[float] = []
    details: List[Dict[str, Any]] = []

    for gene, a1, a2 in cases:
        base_locus = get_pharmacogene_locus(gene)
        expected_label = "/".join(sorted((a1, a2)))
        expected_cnv = expected_cnv_state_for_diplotype(base_locus, a1, a2)

        case_dip_hits = 0
        case_cnv_hits = 0
        for _ in range(trials_per_case):
            sim_locus, frags, bins, h1, h2 = gen.simulate_sample(
                base_locus, a1, a2, mean_depth=45.0, seq_error_rate=0.012
            )
            phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2), seed=seed)
            cnv_prof = decode_cnv_profile(gene, bins)
            calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv_prof, top_k=3)

            pred_label = calls[0].label
            if pred_label == expected_label:
                diplotype_hits += 1
                case_dip_hits += 1

            # Check CNV classification match
            if expected_cnv == CNVState.HYBRID_CONV:
                cnv_ok = cnv_prof.has_gene_conversion
            elif expected_cnv == CNVState.NEUTRAL_2:
                cnv_ok = abs(cnv_prof.estimated_total_copies - 2.0) < 0.6 and not cnv_prof.has_gene_conversion
            else:
                cnv_ok = abs(cnv_prof.estimated_total_copies - expected_cnv.total_copies) < 0.65

            if cnv_ok:
                cnv_hits += 1
                case_cnv_hits += 1

            if phasing.switch_error_rate is not None:
                switch_errors.append(phasing.switch_error_rate)
            total += 1

        details.append(
            {
                "gene": gene,
                "true_diplotype": expected_label,
                "expected_cnv": expected_cnv.value,
                "diplotype_recall": f"{case_dip_hits}/{trials_per_case}",
                "cnv_recall": f"{case_cnv_hits}/{trials_per_case}",
            }
        )

    mean_switch = sum(switch_errors) / len(switch_errors) if switch_errors else 0.0
    return {
        "total_trials": total,
        "diplotype_accuracy": diplotype_hits / max(1, total),
        "cnv_accuracy": cnv_hits / max(1, total),
        "mean_switch_error_rate": mean_switch,
        "cases": details,
    }
