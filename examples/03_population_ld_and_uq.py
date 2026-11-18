"""Example 03: Population HWE exact tests, EM haplotype frequencies, LD heatmap & Monte Carlo UQ."""

from __future__ import annotations

from allele_weaver.catalog import (
    SyntheticCohortGenerator,
    get_drug_pk_params,
    get_pharmacogene_locus,
)
from allele_weaver.cnv_hmm import decode_cnv_profile
from allele_weaver.models import DosingRegimen
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.popgen_uq import analyze_population_ld, propagate_diplotype_to_pk_uq
from allele_weaver.star_caller import call_star_alleles
from allele_weaver.visualizer import render_ld_heatmap


def main() -> None:
    locus = get_pharmacogene_locus("CYP2D6")
    gen = SyntheticCohortGenerator(seed=77)
    panel = gen.simulate_population_panel(locus, n_individuals=250)
    ld = analyze_population_ld(panel)

    print(f"CYP2D6 Population Panel (N=250): EM converged in {ld.em_iterations} iterations")
    for rsid, pval in zip(ld.rsids, ld.hwe_p_values):
        print(f"  HWE {rsid:<11s} p={pval:.4f}")
    print("\nTop EM Haplotype Frequencies:")
    for hap, freq in ld.em_haplotypes[:6]:
        print(f"  {hap} : {freq:.4f}")
    print("\n" + render_ld_heatmap(ld))

    # Propagate diplotype uncertainty into codeine active-metabolite exposure
    sim_locus, frags, bins, h1, h2 = gen.simulate_sample(locus, "*1", "*41", mean_depth=35.0)
    phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2))
    cnv = decode_cnv_profile("CYP2D6", bins)
    calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv, top_k=5)

    codeine = get_drug_pk_params("codeine")
    reg = DosingRegimen(dose_mg=30.0, interval_h=8.0, n_doses=6)
    uq = propagate_diplotype_to_pk_uq(codeine, reg, calls, n_draws=60, iiv_cv=0.20, seed=77)
    print(
        f"\nCodeine Morphine AUC90% CI: [{uq.auc_p05:.4f}, {uq.auc_p95:.4f}] mg*h/L "
        f"(median={uq.auc_p50:.4f}, P(subtherapeutic)={uq.prob_subtherapeutic:.2f})"
    )
    print("Normalized PK Elasticity Indices:")
    for k, v in uq.sensitivity_indices.items():
        print(f"  {k:<20s} : {v:+.3f}")


if __name__ == "__main__":
    main()
