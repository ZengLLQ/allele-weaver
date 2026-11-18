"""Hidden Markov Model (HMM) for pharmacogene structural CNV & paralog gene-conversion.

Models 6 structural states across exon/intron genomic bins:
    DEL_0 (0 copies), HET_DEL_1 (1 copy), NEUTRAL_2 (2 copies),
    DUP_3 (3 copies), MULTI_4 (4 copies), HYBRID_CONV (paralog exon conversion)

Joint emissions combine:
1. Negative-Binomial (Gamma-Poisson overdispersed) read-depth log-PMF scaled by
   copy ratio and calibrated diploid baseline depth.
2. Beta-Binomial / Binomial folded B-Allele Frequency (BAF) log-PMF modulated
   by local paralog sequence homology (e.g. CYP2D7P exon-9 conversion).

Decodes maximum a posteriori state paths via log-space Viterbi and marginal
state posteriors via numerically stabilized Forward-Backward scaling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp

from allele_weaver.models import (
    CNVBinObservation,
    CNVEvent,
    CNVPosteriorProfile,
    CNVState,
    NumericalInstabilityError,
)

HMM_STATES: List[CNVState] = [
    CNVState.DEL_0,
    CNVState.HET_DEL_1,
    CNVState.NEUTRAL_2,
    CNVState.DUP_3,
    CNVState.MULTI_4,
    CNVState.HYBRID_CONV,
]

# Expected relative read-depth multiplier (relative to diploid=1.0) per state
_STATE_DEPTH_RATIO: Dict[CNVState, float] = {
    CNVState.DEL_0: 0.06,
    CNVState.HET_DEL_1: 0.50,
    CNVState.NEUTRAL_2: 1.00,
    CNVState.DUP_3: 1.50,
    CNVState.MULTI_4: 2.00,
    CNVState.HYBRID_CONV: 0.65,
}

# Expected folded BAF center per state
_STATE_BAF_MEANS: Dict[CNVState, Tuple[float, ...]] = {
    CNVState.DEL_0: (0.04,),
    CNVState.HET_DEL_1: (0.04,),
    CNVState.NEUTRAL_2: (0.50, 0.03),
    CNVState.DUP_3: (0.333, 0.03),
    CNVState.MULTI_4: (0.25, 0.50, 0.03),
    CNVState.HYBRID_CONV: (0.22,),
}


def _neg_binom_log_pmf(k: float, mu: float, dispersion_r: float = 18.0) -> float:
    """Compute Negative-Binomial log-PMF parameterized by mean mu and shape r."""
    k_safe = max(0.0, float(k))
    mu_safe = max(1e-3, float(mu))
    r = max(1.0, float(dispersion_r))
    p = r / (r + mu_safe)
    return float(
        gammaln(k_safe + r)
        - gammaln(r)
        - gammaln(k_safe + 1.0)
        + r * math.log(p)
        + k_safe * math.log(max(1e-12, 1.0 - p))
    )


def _folded_baf_log_likelihood(
    alt_count: int,
    total_count: int,
    state: CNVState,
    paralog_homology: float,
) -> float:
    """Compute folded B-Allele Frequency log-likelihood for a bin under a CNV state."""
    if total_count <= 0:
        return 0.0
    k = min(alt_count, total_count - alt_count)
    n = total_count
    comb = float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

    centers = _STATE_BAF_MEANS[state]
    comp_lls: List[float] = []
    for p_base in centers:
        # Paralog homology shifts expected low-BAF floor upward slightly
        p = min(0.49, max(0.015, p_base + 0.06 * paralog_homology if p_base < 0.1 else p_base))
        ll = comb + k * math.log(p) + (n - k) * math.log(1.0 - p)
        comp_lls.append(ll)
    # Weight down BAF contribution relative to read depth when counts are noisy
    return 0.45 * float(logsumexp(comp_lls) - math.log(len(comp_lls)))


def compute_bin_emission_log_probs(
    obs: CNVBinObservation,
    baseline_scale: float = 1.0,
    dispersion_r: float = 18.0,
) -> np.ndarray:
    """Compute joint (depth + BAF + paralog) emission log-likelihood vector across HMM_STATES."""
    log_em = np.zeros(len(HMM_STATES), dtype=float)
    base_depth = max(2.0, obs.expected_diploid_depth * baseline_scale)

    for idx, state in enumerate(HMM_STATES):
        ratio = _STATE_DEPTH_RATIO[state]
        # HYBRID_CONV only activates strongly in high-paralog homology regions
        if state == CNVState.HYBRID_CONV and obs.paralog_homology < 0.5:
            paralog_penalty = -6.5
        elif state == CNVState.HYBRID_CONV and obs.paralog_homology >= 0.85:
            paralog_penalty = 1.2
        else:
            paralog_penalty = 0.0

        depth_ll = _neg_binom_log_pmf(obs.observed_depth, base_depth * ratio, dispersion_r=dispersion_r)
        baf_ll = _folded_baf_log_likelihood(
            obs.het_alt_count, obs.het_total_count, state, obs.paralog_homology
        )
        log_em[idx] = depth_ll + baf_ll + paralog_penalty

    return log_em


def build_transition_matrix(stay_prob: float = 0.90) -> np.ndarray:
    """Construct Markov transition matrix over HMM_STATES with structural switch priors."""
    n = len(HMM_STATES)
    if not (0.5 <= stay_prob < 1.0):
        raise NumericalInstabilityError(f"stay_prob must be in [0.5, 1.0), got {stay_prob}")

    mat = np.full((n, n), (1.0 - stay_prob) / (n - 1), dtype=float)
    np.fill_diagonal(mat, stay_prob)

    # Encourage neutral <-> hybrid transitions at exon-9 boundaries over direct DEL_0 jumps
    idx_neutral = HMM_STATES.index(CNVState.NEUTRAL_2)
    idx_hybrid = HMM_STATES.index(CNVState.HYBRID_CONV)
    idx_del0 = HMM_STATES.index(CNVState.DEL_0)
    mat[idx_neutral, idx_hybrid] *= 1.8
    mat[idx_hybrid, idx_neutral] *= 1.8
    mat[idx_del0, idx_hybrid] *= 0.25
    mat /= mat.sum(axis=1, keepdims=True)
    return mat


def calibrate_hmm_baseline(bins: Sequence[CNVBinObservation]) -> float:
    """Estimate multiplicative diploid library-depth correction factor via trimmed median."""
    if not bins:
        raise NumericalInstabilityError("Cannot calibrate HMM baseline on empty bin list")
    ratios = [
        b.observed_depth / max(1.0, b.expected_diploid_depth)
        for b in bins
        if b.expected_diploid_depth > 0.0
    ]
    med = float(np.median(ratios))
    # Only rescale modest library-size drift (within [0.75, 1.35]) without masking true CNVs
    if 0.78 <= med <= 1.28:
        return med
    return 1.0


def viterbi_decode(
    log_emissions: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> Tuple[List[int], float]:
    """Decode globally optimal state sequence in log-space using Viterbi."""
    t_steps, n_states = log_emissions.shape
    dp = np.zeros((t_steps, n_states), dtype=float)
    bp = np.zeros((t_steps, n_states), dtype=int)

    dp[0] = log_init + log_emissions[0]
    for t in range(1, t_steps):
        scores = dp[t - 1][:, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        bp[t] = best_prev
        dp[t] = scores[best_prev, np.arange(n_states)] + log_emissions[t]

    best_last = int(np.argmax(dp[-1]))
    best_log_prob = float(dp[-1, best_last])
    path = [0] * t_steps
    path[-1] = best_last
    for t in range(t_steps - 1, 0, -1):
        path[t - 1] = int(bp[t, path[t]])
    return path, best_log_prob


def forward_backward(
    log_emissions: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute marginal posterior state probabilities P(S_t = s | O) via Forward-Backward.

    All computations remain in log-space using logsumexp to guarantee zero underflow
    and exact row-wise probability normalization (sum == 1.0).
    """
    t_steps, n_states = log_emissions.shape
    log_alpha = np.zeros((t_steps, n_states), dtype=float)
    log_beta = np.zeros((t_steps, n_states), dtype=float)

    log_alpha[0] = log_init + log_emissions[0]
    for t in range(1, t_steps):
        for s in range(n_states):
            log_alpha[t, s] = float(logsumexp(log_alpha[t - 1] + log_trans[:, s])) + log_emissions[t, s]

    log_likelihood = float(logsumexp(log_alpha[-1]))

    log_beta[-1] = 0.0
    for t in range(t_steps - 2, -1, -1):
        for s in range(n_states):
            log_beta[t, s] = float(
                logsumexp(log_trans[s, :] + log_emissions[t + 1, :] + log_beta[t + 1, :])
            )

    log_post = log_alpha + log_beta
    log_norm = np.array([float(logsumexp(row)) for row in log_post])[:, None]
    posteriors = np.exp(log_post - log_norm)
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return posteriors, log_likelihood


def extract_cnv_events(
    bins: Sequence[CNVBinObservation],
    viterbi_path: Sequence[CNVState],
    posteriors: np.ndarray,
) -> List[CNVEvent]:
    """Segment decoded Viterbi path into contiguous structural CNV/hybrid runs."""
    if not bins:
        return []
    events: List[CNVEvent] = []
    start_idx = 0
    cur_state = viterbi_path[0]

    def _finalize(s_idx: int, e_idx: int, state: CNVState) -> CNVEvent:
        state_col = HMM_STATES.index(state)
        mean_ratio = float(
            np.mean([
                bins[i].observed_depth / max(1.0, bins[i].expected_diploid_depth)
                for i in range(s_idx, e_idx + 1)
            ])
        )
        mean_post = float(np.mean(posteriors[s_idx:e_idx + 1, state_col]))
        return CNVEvent(
            state=state,
            start_bin=s_idx,
            end_bin=e_idx,
            start_bp=bins[s_idx].start_bp,
            end_bp=bins[e_idx].end_bp,
            mean_depth_ratio=mean_ratio,
            mean_posterior_prob=mean_post,
        )

    for idx in range(1, len(viterbi_path)):
        if viterbi_path[idx] != cur_state:
            events.append(_finalize(start_idx, idx - 1, cur_state))
            start_idx = idx
            cur_state = viterbi_path[idx]
    events.append(_finalize(start_idx, len(viterbi_path) - 1, cur_state))
    return events


def decode_cnv_profile(
    locus_name: str,
    bins: Sequence[CNVBinObservation],
    stay_prob: float = 0.90,
    calibrate_baseline: bool = False,
) -> CNVPosteriorProfile:
    """Run joint Viterbi + Forward-Backward CNV & hybrid-conversion inference."""
    if not bins:
        raise NumericalInstabilityError(f"Cannot decode CNV profile for {locus_name} with 0 bins")

    scale = calibrate_hmm_baseline(bins) if calibrate_baseline else 1.0
    t_steps = len(bins)
    n_states = len(HMM_STATES)

    log_emissions = np.zeros((t_steps, n_states), dtype=float)
    for t, obs in enumerate(bins):
        log_emissions[t] = compute_bin_emission_log_probs(obs, baseline_scale=scale)

    trans = build_transition_matrix(stay_prob=stay_prob)
    log_trans = np.log(np.maximum(trans, 1e-12))

    init_probs = np.array([0.04, 0.08, 0.72, 0.09, 0.04, 0.03], dtype=float)
    init_probs /= init_probs.sum()
    log_init = np.log(init_probs)

    v_indices, _ = viterbi_decode(log_emissions, log_trans, log_init)
    viterbi_states = [HMM_STATES[i] for i in v_indices]
    posteriors, log_lik = forward_backward(log_emissions, log_trans, log_init)

    events = extract_cnv_events(bins, viterbi_states, posteriors)

    # Expected diploid copy count from posterior expectation across bins
    state_copies = np.array([s.total_copies for s in HMM_STATES], dtype=float)
    per_bin_copies = posteriors @ state_copies
    # Check non-hybrid core bins if a conversion event occurred at the 3' tail
    has_conv = any(s == CNVState.HYBRID_CONV for s in viterbi_states)
    est_copies = float(np.median(per_bin_copies))

    return CNVPosteriorProfile(
        locus_name=locus_name,
        bins=list(bins),
        states=HMM_STATES,
        viterbi_path=viterbi_states,
        posterior_matrix=posteriors.tolist(),
        log_likelihood=log_lik,
        events=events,
        estimated_total_copies=est_copies,
        has_gene_conversion=has_conv,
    )
