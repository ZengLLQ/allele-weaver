"""Bayesian combinatorial star-allele diplotype inference and CPIC phenotyping.

Evaluates all unordered diploid star-allele pairs (A_i, A_j) for a pharmacogene
locus by fusing four evidence streams in log-probability space:
1. Site-level Binomial read-depth allele likelihood adjusted for copy-number
   allelic dosage (c_1 * x_1 + c_2 * x_2) / (c_1 + c_2) and paralogous
   pseudogene cross-mapping rate rho_paralog.
2. Read-backed phased haplotype likelihood: compares predicted haplotypes
   (H_0, H_1) across all MEC-phased blocks against the pair's defining variants
   under both phase orientations, resolving cis/trans ambiguity (e.g. *1/*4
   vs *10/*something).
3. Structural CNV HMM likelihood: compares (c_1 + c_2) and hybrid conversion
   flags against the HMM posterior distribution across bins.
4. Hardy-Weinberg population prior derived from curated star-allele frequencies.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp

from allele_weaver.cnv_hmm import HMM_STATES
from allele_weaver.models import (
    CNVPosteriorProfile,
    CNVState,
    DiplotypeCall,
    MetabolizerPhenotype,
    PharmacogeneLocus,
    PhasingResult,
    StarAlleleDefinition,
)


def classify_metabolizer_phenotype(gene_symbol: str, activity_score: float) -> MetabolizerPhenotype:
    """Map CPIC gene-specific Activity Score (AS) to standardized phenotype."""
    g = gene_symbol.strip().upper()
    if g == "CYP2D6":
        if activity_score <= 0.0:
            return MetabolizerPhenotype.POOR
        if activity_score < 1.25:
            return MetabolizerPhenotype.INTERMEDIATE
        if activity_score <= 2.25:
            return MetabolizerPhenotype.NORMAL
        return MetabolizerPhenotype.ULTRARAPID

    if g == "CYP2C19":
        if activity_score <= 0.5:
            return MetabolizerPhenotype.POOR
        if activity_score < 1.75:
            return MetabolizerPhenotype.INTERMEDIATE
        if activity_score <= 2.25:
            return MetabolizerPhenotype.NORMAL
        if activity_score < 2.75:
            return MetabolizerPhenotype.RAPID
        return MetabolizerPhenotype.ULTRARAPID

    if g in ("CYP2C9", "DPYD"):
        if activity_score <= 0.5:
            return MetabolizerPhenotype.POOR
        if activity_score < 1.75:
            return MetabolizerPhenotype.INTERMEDIATE
        return MetabolizerPhenotype.NORMAL

    if activity_score <= 0.5:
        return MetabolizerPhenotype.POOR
    if activity_score < 1.5:
        return MetabolizerPhenotype.INTERMEDIATE
    if activity_score <= 2.25:
        return MetabolizerPhenotype.NORMAL
    return MetabolizerPhenotype.ULTRARAPID


def _binomial_log_pmf(k: int, n: int, p: float) -> float:
    """Compute exact log Binomial(k | n, p) with safe probability clamping."""
    if n <= 0:
        return 0.0
    p_safe = min(1.0 - 1e-5, max(1e-5, p))
    comb = float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
    return comb + k * math.log(p_safe) + (n - k) * math.log(1.0 - p_safe)


def _allele_vector(locus: PharmacogeneLocus, star: StarAlleleDefinition) -> List[int]:
    """Return binary variant vector across locus.variants for one star allele."""
    if star.copy_count == 0:
        return [0] * len(locus.variants)
    return [star.defining_variants.get(v.rsid, 0) for v in locus.variants]


def _score_site_dosage_ll(
    locus: PharmacogeneLocus,
    s1: StarAlleleDefinition,
    s2: StarAlleleDefinition,
    seq_error_floor: float = 0.015,
) -> float:
    """Compute Binomial log-likelihood of variant alt/ref depths under (s1, s2)."""
    vec1 = _allele_vector(locus, s1)
    vec2 = _allele_vector(locus, s2)
    c1, c2 = s1.copy_count, s2.copy_count
    tot_c = c1 + c2

    ll = 0.0
    for idx, var in enumerate(locus.variants):
        if tot_c == 0:
            raw_vaf = 0.0
        else:
            raw_vaf = (c1 * vec1[idx] + c2 * vec2[idx]) / tot_c

        # Incorporate paralogous cross-mapping bias (e.g. CYP2D7 rs1135840 G leakage)
        paralog_target = 1.0 if var.rsid == "rs1135840" else 0.0
        eff_vaf = (1.0 - var.paralog_mismap_rate) * raw_vaf + var.paralog_mismap_rate * paralog_target
        # Fold in sequencing error floor
        obs_p = eff_vaf * (1.0 - seq_error_floor) + (1.0 - eff_vaf) * seq_error_floor
        ll += _binomial_log_pmf(var.alt_count, var.total_depth, obs_p)
    return ll


def _score_phased_consistency(
    locus: PharmacogeneLocus,
    s1: StarAlleleDefinition,
    s2: StarAlleleDefinition,
    phasing: Optional[PhasingResult],
) -> Tuple[float, bool]:
    """Evaluate phase-block agreement log-likelihood and boolean cis/trans consistency.

    Computes max(LL(H0~s1, H1~s2), LL(H0~s2, H1~s1)) per phased block so global
    haplotype swap never penalizes a true diplotype, while cis vs trans errors
    (e.g. two variants in cis vs trans) receive a decisive log-odds penalty.
    """
    if phasing is None or not phasing.blocks:
        return 0.0, True

    if s1.copy_count == 0 or s2.copy_count == 0:
        # Homozygous/hemizygous deletions have no heterozygous phase blocks
        return 0.0, True

    vec1 = _allele_vector(locus, s1)
    vec2 = _allele_vector(locus, s2)

    total_bonus = 0.0
    all_consistent = True

    for block in phasing.blocks:
        # Only score informative positions where the two alleles differ or variant is ALT
        direct_ll = 0.0
        swap_ll = 0.0
        direct_mismatch = 0
        swap_mismatch = 0

        for local_i, site_idx in enumerate(block.variant_indices):
            h0 = block.haplotype_0[local_i]
            h1 = block.haplotype_1[local_i]
            v1 = vec1[site_idx]
            v2 = vec2[site_idx]
            # Only apply phase-orientation scoring at sites with heterozygous calls
            if h0 == h1 and v1 == v2:
                continue
            conf = max(1.0, min(8.0, block.site_confidence[local_i]))

            # Direct orientation: H0 -> s1, H1 -> s2
            if h0 == v1 and h1 == v2:
                direct_ll += 0.45 * conf
            else:
                direct_mismatch += 1
                direct_ll -= 1.2 * conf

            # Swapped orientation: H0 -> s2, H1 -> s1
            if h0 == v2 and h1 == v1:
                swap_ll += 0.45 * conf
            else:
                swap_mismatch += 1
                swap_ll -= 1.2 * conf

        best_ll = max(direct_ll, swap_ll)
        best_mismatch = direct_mismatch if direct_ll >= swap_ll else swap_mismatch
        total_bonus += best_ll
        if best_mismatch > 0:
            all_consistent = False

    return total_bonus, all_consistent


def _score_cnv_consistency(
    s1: StarAlleleDefinition,
    s2: StarAlleleDefinition,
    cnv_profile: Optional[CNVPosteriorProfile],
) -> Tuple[float, bool]:
    """Score structural copy-count & hybrid conversion compatibility with CNV HMM."""
    if cnv_profile is None or not cnv_profile.posterior_matrix:
        return 0.0, True

    tot_copies = s1.copy_count + s2.copy_count
    pair_has_conv = (
        s1.structural_type == "hybrid_conversion" or s2.structural_type == "hybrid_conversion"
    )

    if tot_copies == 0:
        target_state = CNVState.DEL_0
    elif tot_copies == 1:
        target_state = CNVState.HET_DEL_1
    elif tot_copies == 3:
        target_state = CNVState.DUP_3
    elif tot_copies >= 4:
        target_state = CNVState.MULTI_4
    else:
        target_state = CNVState.NEUTRAL_2

    target_idx = HMM_STATES.index(target_state)
    hybrid_idx = HMM_STATES.index(CNVState.HYBRID_CONV)
    post = np.asarray(cnv_profile.posterior_matrix, dtype=float)

    n_bins = post.shape[0]
    ll = 0.0
    if pair_has_conv:
        # Expect core 5' bins in target_state (or neutral/het) and 3' tail in HYBRID_CONV
        split = max(1, n_bins - 3)
        core_probs = np.clip(post[:split, target_idx], 1e-6, 1.0)
        tail_probs = np.clip(post[split:, hybrid_idx], 1e-6, 1.0)
        ll += float(np.sum(np.log(core_probs)) + 2.0 * np.sum(np.log(tail_probs)))
        consistent = cnv_profile.has_gene_conversion
    else:
        bin_probs = np.clip(post[:, target_idx], 1e-6, 1.0)
        ll += float(np.sum(np.log(bin_probs)))
        if cnv_profile.has_gene_conversion:
            ll -= 8.0
        consistent = (
            abs(tot_copies - cnv_profile.estimated_total_copies) <= 0.65
            and not cnv_profile.has_gene_conversion
        )

    return 1.5 * ll, consistent


def call_star_alleles(
    locus: PharmacogeneLocus,
    phasing: Optional[PhasingResult] = None,
    cnv_profile: Optional[CNVPosteriorProfile] = None,
    top_k: int = 8,
) -> List[DiplotypeCall]:
    """Enumerate and rank candidate star-allele diplotypes by Bayesian posterior.

    Guarantees posterior_prob values across the returned full normalized space
    sum to 1.0 (returned list truncated to top_k after full normalization).
    """
    allele_names = sorted(locus.star_alleles.keys())
    n_alleles = len(allele_names)

    freq_sum = sum(a.population_frequency for a in locus.star_alleles.values())
    norm_freqs = {k: v.population_frequency / freq_sum for k, v in locus.star_alleles.items()}

    candidates: List[Tuple[str, str, float, float, float, float, bool, bool]] = []

    for i in range(n_alleles):
        for j in range(i, n_alleles):
            a1_name = allele_names[i]
            a2_name = allele_names[j]
            s1 = locus.star_alleles[a1_name]
            s2 = locus.star_alleles[a2_name]

            # Hardy-Weinberg diploid prior: p^2 for homozygote, 2*p*q for heterozygote
            p1 = norm_freqs[a1_name]
            p2 = norm_freqs[a2_name]
            hw_prior = (p1 * p2) if i == j else (2.0 * p1 * p2)
            log_prior = math.log(max(1e-12, hw_prior))

            site_ll = _score_site_dosage_ll(locus, s1, s2)
            phase_ll, phase_ok = _score_phased_consistency(locus, s1, s2, phasing)
            cnv_ll, cnv_ok = _score_cnv_consistency(s1, s2, cnv_profile)

            total_ll = site_ll + phase_ll + cnv_ll
            log_post_unnorm = total_ll + log_prior
            act_score = s1.activity_value + s2.activity_value

            candidates.append(
                (a1_name, a2_name, log_post_unnorm, total_ll, log_prior, act_score, phase_ok, cnv_ok)
            )

    log_evidences = np.array([c[2] for c in candidates], dtype=float)
    log_z = float(logsumexp(log_evidences))

    calls: List[DiplotypeCall] = []
    for a1_name, a2_name, log_post_u, ll, lp, act, phase_ok, cnv_ok in candidates:
        post_prob = math.exp(log_post_u - log_z)
        pheno = classify_metabolizer_phenotype(locus.gene_symbol, act)
        notes: List[str] = []
        if phase_ok:
            notes.append("phase-concordant")
        else:
            notes.append("phase-discordant")
        if cnv_ok:
            notes.append("cnv-concordant")
        else:
            notes.append("cnv-discordant")

        calls.append(
            DiplotypeCall(
                allele_1=a1_name,
                allele_2=a2_name,
                posterior_prob=post_prob,
                log_likelihood=ll,
                log_prior=lp,
                activity_score=act,
                phenotype=pheno,
                phase_consistent=phase_ok,
                cnv_consistent=cnv_ok,
                supporting_notes=tuple(notes),
            )
        )

    calls.sort(key=lambda c: c.posterior_prob, reverse=True)
    return calls[: max(1, top_k)]
