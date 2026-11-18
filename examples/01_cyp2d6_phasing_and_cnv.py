"""Example 01: CYP2D6 read-backed MEC phasing, CYP2D7P hybrid CNV HMM, and star-allele calling."""

from __future__ import annotations

from allele_weaver.catalog import SyntheticCohortGenerator, get_pharmacogene_locus
from allele_weaver.cnv_hmm import decode_cnv_profile
from allele_weaver.phasing import phase_locus_reads
from allele_weaver.star_caller import call_star_alleles
from allele_weaver.visualizer import (
    render_cnv_profile,
    render_diplotype_posteriors,
    render_haplotype_tracks,
)


def main() -> None:
    locus = get_pharmacogene_locus("CYP2D6")
    gen = SyntheticCohortGenerator(seed=101)

    for true_a1, true_a2 in [("*1", "*4"), ("*10", "*36"), ("*1", "*1x2")]:
        sim_locus, frags, bins, h1, h2 = gen.simulate_sample(
            locus, true_a1, true_a2, mean_depth=50.0, seq_error_rate=0.012
        )
        phasing = phase_locus_reads(sim_locus, frags, truth_haplotypes=(h1, h2), seed=101)
        cnv_prof = decode_cnv_profile("CYP2D6", bins)
        calls = call_star_alleles(sim_locus, phasing=phasing, cnv_profile=cnv_prof, top_k=4)

        print(f"\n>>> Simulated Ground Truth: {true_a1}/{true_a2}")
        print(render_haplotype_tracks(sim_locus, phasing))
        print(render_cnv_profile(cnv_prof))
        print(render_diplotype_posteriors(calls))


if __name__ == "__main__":
    main()
