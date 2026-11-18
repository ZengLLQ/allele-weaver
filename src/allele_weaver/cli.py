"""Command-line interface for AlleleWeaver pharmacogenomics & dosing simulation.

Provides 6 operational subcommands:
    phase     - Read-backed Minimum Error Correction (MEC) haplotype phasing
    call      - Paralog-aware CNV HMM + Bayesian star-allele diplotype calling
    dose      - Multi-compartment RKF45 PBPK/PD simulation & dose optimization
    popgen    - Population HWE exact test, LD matrix & EM haplotype estimation
    benchmark - Multi-gene synthetic accuracy & switch-error benchmark suite
    demo      - End-to-end clinical pharmacogenomics pipeline & ASCII dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from allele_weaver.catalog import (
    SyntheticCohortGenerator,
    get_drug_pk_params,
    get_pharmacogene_locus,
)
from allele_weaver.cnv_hmm import decode_cnv_profile
from allele_weaver.io_formats import (
    export_clinical_json,
    export_pk_telemetry_csv,
    parse_fragment_tsv,
    parse_vcf,
)
from allele_weaver.models import DosingRegimen, PharmacogeneLocus
from allele_weaver.pbpk_dosing import optimize_precision_dose, simulate_pbpk_regimen
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.pipeline import PharmacogenomicsPipeline, run_benchmark_suite
from allele_weaver.popgen_uq import analyze_population_ld
from allele_weaver.star_caller import call_star_alleles
from allele_weaver.visualizer import (
    render_cnv_profile,
    render_diplotype_posteriors,
    render_haplotype_tracks,
    render_ld_heatmap,
    render_pk_curve,
)


def _safe_print(text: str) -> None:
    """Print strictly 7-bit ASCII text so Windows cp1252 consoles never fail."""
    ascii_text = text.encode("ascii", errors="replace").decode("ascii")
    print(ascii_text)


def _load_or_simulate_sample(
    gene: str,
    diplotype: str,
    vcf_path: Optional[str],
    fragments_path: Optional[str],
    depth: float,
    seed: int,
):
    base_locus = get_pharmacogene_locus(gene)
    a1, a2 = [x.strip() for x in diplotype.split("/", 1)]
    gen = SyntheticCohortGenerator(seed=seed)
    sim_locus, sim_frags, sim_bins, h1, h2 = gen.simulate_sample(
        base_locus, a1, a2, mean_depth=depth
    )

    if vcf_path:
        sites = parse_vcf(Path(vcf_path).read_text(encoding="utf-8"))
        sim_locus = PharmacogeneLocus(
            gene_symbol=base_locus.gene_symbol,
            chrom=base_locus.chrom,
            start_bp=base_locus.start_bp,
            end_bp=base_locus.end_bp,
            variants=sites,
            star_alleles=base_locus.star_alleles,
            paralog_regions=base_locus.paralog_regions,
            reference_activity_score=base_locus.reference_activity_score,
            exon_bin_bounds=base_locus.exon_bin_bounds,
        )
    if fragments_path:
        sim_frags = parse_fragment_tsv(Path(fragments_path).read_text(encoding="utf-8"))

    return sim_locus, sim_frags, sim_bins, (h1, h2)


def _cmd_phase(args: argparse.Namespace) -> int:
    locus, frags, _, truth = _load_or_simulate_sample(
        args.gene, args.diplotype, args.vcf, args.fragments, args.depth, args.seed
    )
    res = phase_locus_reads(locus, frags, truth_haplotypes=truth, seed=args.seed)
    _safe_print(render_haplotype_tracks(locus, res))
    if res.switch_error_rate is not None:
        _safe_print(
            f"Switch Error Rate: {res.switch_error_rate:.4f} | "
            f"Hamming Error Rate: {res.hamming_error_rate:.4f}"
        )
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    locus, frags, bins, truth = _load_or_simulate_sample(
        args.gene, args.diplotype, args.vcf, args.fragments, args.depth, args.seed
    )
    phasing = phase_locus_reads(locus, frags, truth_haplotypes=truth, seed=args.seed)
    cnv_prof = decode_cnv_profile(locus.gene_symbol, bins)
    calls = call_star_alleles(locus, phasing=phasing, cnv_profile=cnv_prof, top_k=args.top_k)

    _safe_print(render_cnv_profile(cnv_prof))
    _safe_print("")
    _safe_print(render_diplotype_posteriors(calls))
    if args.json_out:
        payload = {
            "gene": locus.gene_symbol,
            "top_call": calls[0].label,
            "posterior": calls[0].posterior_prob,
            "activity_score": calls[0].activity_score,
            "phenotype": calls[0].phenotype.value,
            "cnv_copies": cnv_prof.estimated_total_copies,
        }
        export_clinical_json(payload, Path(args.json_out))
    return 0


def _cmd_dose(args: argparse.Namespace) -> int:
    drug = get_drug_pk_params(args.drug)
    if args.optimize:
        opt = optimize_precision_dose(
            drug,
            activity_score=args.activity_score,
            diplotype_label=args.label,
            nominal_dose_mg=args.dose_mg,
        )
        _safe_print(
            f"Optimized Regimen for {drug.drug_name} (AS={args.activity_score:.2f}): "
            f"{opt.optimal_dose_mg:.2f} mg q{opt.optimal_interval_h:.0f}h | "
            f"Feasible={opt.feasible} | SafetyMargin={opt.safety_margin_ratio:.2f}x"
        )
        sim = opt.simulation
    else:
        reg = DosingRegimen(dose_mg=args.dose_mg, interval_h=args.interval_h, n_doses=args.n_doses)
        sim = simulate_pbpk_regimen(
            drug, reg, activity_score=args.activity_score, diplotype_label=args.label
        )

    _safe_print(render_pk_curve(sim, drug))
    _safe_print(
        f"RKF45 Steps: {sim.rkf_steps_taken} accepted, {sim.rkf_steps_rejected} rejected | "
        f"Mass-Balance Error: {sim.max_mass_balance_error_mg:.3e} mg"
    )
    if args.csv_out:
        export_pk_telemetry_csv(sim, Path(args.csv_out))
    return 0


def _cmd_popgen(args: argparse.Namespace) -> int:
    locus = get_pharmacogene_locus(args.gene)
    gen = SyntheticCohortGenerator(seed=args.seed)
    panel = gen.simulate_population_panel(locus, n_individuals=args.n_samples)
    ld = analyze_population_ld(panel)

    _safe_print(
        f"=== POPULATION PHARMACOGENETICS: {locus.gene_symbol} (N={args.n_samples}) ==="
    )
    _safe_print(f"EM Converged in {ld.em_iterations} iterations (LL={ld.em_log_likelihood:.2f})")
    _safe_print("Site HWE Exact p-values:")
    for rsid, pval in zip(ld.rsids, ld.hwe_p_values):
        _safe_print(f"  {rsid:<12s}: p={pval:.4f}")
    _safe_print("")
    _safe_print("Top EM Haplotype Frequencies:")
    for hap, freq in ld.em_haplotypes[:6]:
        _safe_print(f"  {hap} : {freq:.4f}")
    _safe_print("")
    _safe_print(render_ld_heatmap(ld))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark_suite(seed=args.seed, trials_per_case=args.trials)
    _safe_print("=== ALLELEWEAVER MULTI-GENE BENCHMARK SUITE ===")
    _safe_print(
        f"Total Trials: {report['total_trials']} | "
        f"Diplotype Acc: {report['diplotype_accuracy'] * 100:.1f}% | "
        f"CNV Acc: {report['cnv_accuracy'] * 100:.1f}% | "
        f"Mean Switch Err: {report['mean_switch_error_rate']:.4f}"
    )
    for row in report["cases"]:
        _safe_print(
            f"  {row['gene']:<8s} | {row['true_diplotype']:<10s} | "
            f"CNV={row['expected_cnv']:<11s} | Dip={row['diplotype_recall']} | "
            f"CNV_Rec={row['cnv_recall']}"
        )
    if args.json_out:
        export_clinical_json(report, Path(args.json_out))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    locus, frags, bins, truth = _load_or_simulate_sample(
        args.gene, args.diplotype, None, None, args.depth, args.seed
    )
    pipe = PharmacogenomicsPipeline(uq_draws=args.uq_draws, seed=args.seed)
    report = pipe.run(
        locus,
        frags,
        bins,
        drug_name=args.drug,
        nominal_dose_mg=args.dose_mg,
        truth_haplotypes=truth,
    )
    _safe_print(report.ascii_dashboard)
    if args.json_out:
        export_clinical_json(report.to_dict(), Path(args.json_out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct top-level argparse parser with 6 pharmacogenomics subcommands."""
    parser = argparse.ArgumentParser(
        prog="allele-weaver",
        description="Clinical pharmacogenomics haplotype phasing, CNV star-allele calling, and PBPK dosing simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_phase = sub.add_parser("phase", help="Run read-backed MEC haplotype phasing")
    p_phase.add_argument("--gene", default="CYP2D6")
    p_phase.add_argument("--diplotype", default="*1/*4")
    p_phase.add_argument("--vcf", default=None)
    p_phase.add_argument("--fragments", default=None)
    p_phase.add_argument("--depth", type=float, default=45.0)
    p_phase.add_argument("--seed", type=int, default=42)
    p_phase.set_defaults(func=_cmd_phase)

    p_call = sub.add_parser("call", help="Run CNV HMM + Bayesian star-allele diplotype calling")
    p_call.add_argument("--gene", default="CYP2D6")
    p_call.add_argument("--diplotype", default="*10/*36")
    p_call.add_argument("--vcf", default=None)
    p_call.add_argument("--fragments", default=None)
    p_call.add_argument("--depth", type=float, default=45.0)
    p_call.add_argument("--top-k", type=int, default=5)
    p_call.add_argument("--seed", type=int, default=42)
    p_call.add_argument("--json-out", default=None)
    p_call.set_defaults(func=_cmd_call)

    p_dose = sub.add_parser("dose", help="Simulate or optimize genotype-guided PBPK/PD regimen")
    p_dose.add_argument("--drug", default="warfarin")
    p_dose.add_argument("--activity-score", type=float, default=1.0)
    p_dose.add_argument("--label", default="*1/*3")
    p_dose.add_argument("--dose-mg", type=float, default=5.0)
    p_dose.add_argument("--interval-h", type=float, default=24.0)
    p_dose.add_argument("--n-doses", type=int, default=8)
    p_dose.add_argument("--optimize", action="store_true")
    p_dose.add_argument("--csv-out", default=None)
    p_dose.set_defaults(func=_cmd_dose)

    p_pop = sub.add_parser("popgen", help="Compute HWE exact test, LD matrix, and EM haplotypes")
    p_pop.add_argument("--gene", default="CYP2D6")
    p_pop.add_argument("--n-samples", type=int, default=250)
    p_pop.add_argument("--seed", type=int, default=42)
    p_pop.set_defaults(func=_cmd_popgen)

    p_bench = sub.add_parser("benchmark", help="Run multi-gene accuracy and switch-error benchmark")
    p_bench.add_argument("--trials", type=int, default=3)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--json-out", default=None)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_demo = sub.add_parser("demo", help="Run end-to-end genomic-to-PBPK clinical dashboard")
    p_demo.add_argument("--gene", default="CYP2C9")
    p_demo.add_argument("--diplotype", default="*1/*3")
    p_demo.add_argument("--drug", default="warfarin")
    p_demo.add_argument("--dose-mg", type=float, default=5.0)
    p_demo.add_argument("--depth", type=float, default=45.0)
    p_demo.add_argument("--uq-draws", type=int, default=40)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--json-out", default=None)
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
