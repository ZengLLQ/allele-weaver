"""Read-backed haplotype phasing via Minimum Error Correction (MEC) optimization.

Implements a two-tier MEC solver over read-fragment allele matrices:
1. Connected-component decomposition of the read-variant bipartite graph so
   unbridged gaps never produce spurious long-range phasing claims.
2. Exact bitmask dynamic programming (WhatsHap-style) over active-read
   partitions when local spanning coverage k <= exact_dp_max_reads, yielding
   provably optimal MEC partitions.
3. Simulated-annealing + single-read flip refinement for high-coverage blocks,
   minimizing Phred-weighted mismatch penalties while preserving homozygous
   sites (0/0 or 1/1) without artificial bipartition bias.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from allele_weaver.models import (
    PhasedBlock,
    PhasingConvergenceError,
    PhasingResult,
    PharmacogeneLocus,
    ReadFragment,
    phred_to_error_prob,
)


def _read_weight(phred_q: int, mapq: int) -> float:
    """Compute Phred-calibrated log-odds penalty for flipping one allele call."""
    p_err = phred_to_error_prob(phred_q)
    p_map_ok = max(0.5, 1.0 - phred_to_error_prob(max(10, mapq)))
    return p_map_ok * math.log((1.0 - p_err) / p_err)


def _infer_site_zygosities(
    locus: PharmacogeneLocus,
    fragments: Sequence[ReadFragment],
) -> List[int]:
    """Infer site zygosity (0=hom_ref, 1=het, 2=hom_alt) from reads & site depth."""
    n_sites = len(locus.variants)
    frag_ref = [0] * n_sites
    frag_alt = [0] * n_sites
    for frag in fragments:
        for s_idx, (allele, _q) in frag.observations.items():
            if 0 <= s_idx < n_sites:
                if allele == 1:
                    frag_alt[s_idx] += 1
                else:
                    frag_ref[s_idx] += 1

    zygosities: List[int] = []
    for s_idx, var in enumerate(locus.variants):
        f_tot = frag_ref[s_idx] + frag_alt[s_idx]
        # Prefer actual fragment counts when site counts are unpopulated defaults
        if f_tot > 0 and (var.alt_count == 0 or var.total_depth == 0):
            rc, ac = frag_ref[s_idx], frag_alt[s_idx]
        else:
            rc, ac = var.ref_count, var.alt_count

        tot = rc + ac
        if tot == 0:
            zygosities.append(1)
            continue
        vaf = ac / tot
        if tot >= 6 and (ac <= 1 or vaf <= 0.14):
            zygosities.append(0)
        elif tot >= 6 and (rc <= 1 or vaf >= 0.86):
            zygosities.append(2)
        elif ac == 0:
            zygosities.append(0)
        elif rc == 0:
            zygosities.append(2)
        else:
            zygosities.append(1)
    return zygosities


def extract_phase_components(
    n_sites: int,
    fragments: Sequence[ReadFragment],
) -> List[List[int]]:
    """Partition variant indices into connected phasing components via BFS."""
    adj: Dict[int, set[int]] = defaultdict(set)
    seen_sites: set[int] = set()

    for frag in fragments:
        valid_sites = sorted(idx for idx in frag.observations if 0 <= idx < n_sites)
        for idx in valid_sites:
            seen_sites.add(idx)
        if len(valid_sites) >= 2:
            anchor = valid_sites[0]
            for other in valid_sites[1:]:
                adj[anchor].add(other)
                adj[other].add(anchor)

    visited: set[int] = set()
    components: List[List[int]] = []

    for site in sorted(seen_sites):
        if site in visited or not adj[site]:
            continue
        comp: List[int] = []
        queue = deque([site])
        visited.add(site)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(sorted(comp))

    return components


def _column_mec_cost(
    site_obs: List[Tuple[int, int, float]],
    read_partition: Sequence[int],
    h0_bit: int,
    h1_bit: int,
) -> Tuple[float, int]:
    """Compute weighted MEC cost and raw flips at one column given (h0_bit, h1_bit)."""
    cost = 0.0
    flips = 0
    for r_id, allele, weight in site_obs:
        expected = h0_bit if read_partition[r_id] == 0 else h1_bit
        if allele != expected:
            cost += weight
            flips += 1
    return cost, flips


def _solve_exact_dp_component(
    comp_sites: List[int],
    comp_reads: List[Tuple[int, ReadFragment]],
    site_zygosities: Sequence[int],
) -> Tuple[List[int], List[int], List[float], float, int]:
    """Exact column-wise bitmask DP over active reads for MEC optimality."""
    n_cols = len(comp_sites)
    site_to_col = {s: c for c, s in enumerate(comp_sites)}

    read_col_spans: List[Tuple[int, int, Dict[int, Tuple[int, float]]]] = []
    for _, frag in comp_reads:
        cols_obs: Dict[int, Tuple[int, float]] = {}
        for s_idx, (allele, qual) in frag.observations.items():
            if s_idx in site_to_col:
                cols_obs[site_to_col[s_idx]] = (allele, _read_weight(qual, frag.mapping_quality))
        if len(cols_obs) >= 2:
            c_min, c_max = min(cols_obs), max(cols_obs)
            read_col_spans.append((c_min, c_max, cols_obs))

    active_per_col: List[List[int]] = [[] for _ in range(n_cols)]
    for r_idx, (c_min, c_max, _) in enumerate(read_col_spans):
        for c in range(c_min, c_max + 1):
            active_per_col[c].append(r_idx)

    prev_active: List[int] = []
    dp: Dict[int, float] = {0: 0.0}
    backpointers: List[Dict[int, Tuple[int, int, int]]] = []

    for c in range(n_cols):
        cur_active = active_per_col[c]
        if len(cur_active) > 12:
            raise PhasingConvergenceError(
                f"Active read count {len(cur_active)} exceeds exact DP limit 12"
            )

        prev_pos = {r: i for i, r in enumerate(prev_active)}
        shared = [(i, prev_pos[r]) for i, r in enumerate(cur_active) if r in prev_pos]
        new_only = [i for i, r in enumerate(cur_active) if r not in prev_pos]

        col_obs: List[Tuple[int, int, float]] = []
        for bit_pos, r_idx in enumerate(cur_active):
            obs_map = read_col_spans[r_idx][2]
            if c in obs_map:
                allele, weight = obs_map[c]
                col_obs.append((bit_pos, allele, weight))

        best_prev_for_proj: Dict[int, Tuple[float, int]] = {}
        for p_mask, p_cost in dp.items():
            proj = 0
            for cur_i, prev_i in shared:
                if (p_mask >> prev_i) & 1:
                    proj |= 1 << cur_i
            if proj not in best_prev_for_proj or p_cost < best_prev_for_proj[proj][0]:
                best_prev_for_proj[proj] = (p_cost, p_mask)

        new_dp: Dict[int, float] = {}
        col_bp: Dict[int, Tuple[int, int, int]] = {}
        n_new = len(new_only)
        zyg = site_zygosities[comp_sites[c]]

        for proj_mask, (base_cost, orig_prev_mask) in best_prev_for_proj.items():
            for new_assign in range(1 << n_new):
                cur_mask = proj_mask
                for k_idx, cur_bit_pos in enumerate(new_only):
                    if (new_assign >> k_idx) & 1:
                        cur_mask |= 1 << cur_bit_pos

                if zyg == 0:
                    step_cost = sum(w for _, a, w in col_obs if a != 0)
                    h0_choice, h1_choice = 0, 0
                elif zyg == 2:
                    step_cost = sum(w for _, a, w in col_obs if a != 1)
                    h0_choice, h1_choice = 1, 1
                else:
                    cost_01 = 0.0
                    cost_10 = 0.0
                    for bit_pos, allele, weight in col_obs:
                        assign = (cur_mask >> bit_pos) & 1
                        if allele != assign:
                            cost_01 += weight
                        if allele != (1 - assign):
                            cost_10 += weight
                    if cost_01 <= cost_10:
                        step_cost, h0_choice, h1_choice = cost_01, 0, 1
                    else:
                        step_cost, h0_choice, h1_choice = cost_10, 1, 0

                tot_cost = base_cost + step_cost
                if cur_mask not in new_dp or tot_cost < new_dp[cur_mask]:
                    new_dp[cur_mask] = tot_cost
                    col_bp[cur_mask] = (orig_prev_mask, h0_choice, h1_choice)

        dp = new_dp
        backpointers.append(col_bp)
        prev_active = cur_active

    best_mask = min(dp, key=lambda m: dp[m])
    total_cost = dp[best_mask]

    h0: List[int] = [0] * n_cols
    h1: List[int] = [0] * n_cols
    confidence: List[float] = [0.0] * n_cols

    cur_mask = best_mask
    chosen_masks = [0] * n_cols
    for c in range(n_cols - 1, -1, -1):
        chosen_masks[c] = cur_mask
        prev_mask, h0_bit, h1_bit = backpointers[c][cur_mask]
        h0[c] = h0_bit
        h1[c] = h1_bit
        cur_mask = prev_mask

    total_flips = 0
    for c in range(n_cols):
        cur_active = active_per_col[c]
        mask = chosen_masks[c]
        support_chosen = 0.0
        support_alt = 0.0
        for bit_pos, r_idx in enumerate(cur_active):
            obs_map = read_col_spans[r_idx][2]
            if c not in obs_map:
                continue
            allele, weight = obs_map[c]
            assign = (mask >> bit_pos) & 1
            exp_chosen = h0[c] if assign == 0 else h1[c]
            if allele == exp_chosen:
                support_chosen += weight
            else:
                support_alt += weight
                total_flips += 1
        confidence[c] = max(0.0, support_chosen - support_alt)

    return h0, h1, confidence, total_cost, total_flips


def _solve_annealed_component(
    comp_sites: List[int],
    comp_reads: List[Tuple[int, ReadFragment]],
    site_zygosities: Sequence[int],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[float], float, int]:
    """Simulated-annealing + coordinate-descent MEC solver for arbitrary read depth."""
    rng = np.random.default_rng(seed)
    site_to_col = {s: c for c, s in enumerate(comp_sites)}
    n_cols = len(comp_sites)
    col_zygs = [site_zygosities[s] for s in comp_sites]

    read_obs: List[List[Tuple[int, int, float]]] = []
    col_to_reads: List[List[Tuple[int, int, float]]] = [[] for _ in range(n_cols)]

    for _, frag in comp_reads:
        r_entries: List[Tuple[int, int, float]] = []
        for s_idx, (allele, qual) in frag.observations.items():
            if s_idx in site_to_col:
                col = site_to_col[s_idx]
                w = _read_weight(qual, frag.mapping_quality)
                r_entries.append((col, allele, w))
        if r_entries:
            r_id = len(read_obs)
            read_obs.append(r_entries)
            for col, allele, w in r_entries:
                col_to_reads[col].append((r_id, allele, w))

    n_reads = len(read_obs)
    if n_reads == 0:
        return [0] * n_cols, [1] * n_cols, [0.0] * n_cols, 0.0, 0

    def eval_col(col: int, part: Sequence[int]) -> Tuple[int, int, float, int, float]:
        z = col_zygs[col]
        if z == 0:
            c00, f00 = _column_mec_cost(col_to_reads[col], part, 0, 0)
            c11, _ = _column_mec_cost(col_to_reads[col], part, 1, 1)
            return 0, 0, c00, f00, max(0.0, c11 - c00)
        if z == 2:
            c11, f11 = _column_mec_cost(col_to_reads[col], part, 1, 1)
            c00, _ = _column_mec_cost(col_to_reads[col], part, 0, 0)
            return 1, 1, c11, f11, max(0.0, c00 - c11)
        c01, f01 = _column_mec_cost(col_to_reads[col], part, 0, 1)
        c10, f10 = _column_mec_cost(col_to_reads[col], part, 1, 0)
        if c01 <= c10:
            return 0, 1, c01, f01, c10 - c01
        return 1, 0, c10, f10, c01 - c10

    def evaluate_partition(part: np.ndarray) -> Tuple[List[int], List[int], List[float], float, int]:
        h0_out = [0] * n_cols
        h1_out = [0] * n_cols
        conf_out = [0.0] * n_cols
        tot_cost = 0.0
        tot_flips = 0
        part_list = part.tolist()
        for c in range(n_cols):
            b0, b1, cst, flp, cf = eval_col(c, part_list)
            h0_out[c] = b0
            h1_out[c] = b1
            conf_out[c] = cf
            tot_cost += cst
            tot_flips += flp
        return h0_out, h1_out, conf_out, tot_cost, tot_flips

    best_part = np.zeros(n_reads, dtype=int)
    best_cost = float("inf")

    for restart in range(4):
        part = rng.integers(0, 2, size=n_reads, dtype=int)
        part[0] = 0
        h0_cur, h1_cur, _, cur_cost, _ = evaluate_partition(part)

        n_steps = max(80, n_reads * 6)
        t_start = 4.0
        t_end = 0.05
        for step in range(n_steps):
            temp = t_start * ((t_end / t_start) ** (step / max(1, n_steps - 1)))
            r_id = int(rng.integers(1, n_reads)) if n_reads > 1 else 0
            if r_id == 0 and restart == 0:
                continue

            delta = 0.0
            for col, allele, w in read_obs[r_id]:
                if col_zygs[col] != 1:
                    continue
                old_exp = h0_cur[col] if part[r_id] == 0 else h1_cur[col]
                new_exp = h1_cur[col] if part[r_id] == 0 else h0_cur[col]
                delta += (w if allele != new_exp else 0.0) - (w if allele != old_exp else 0.0)

            if delta <= 0.0 or rng.random() < math.exp(-delta / max(1e-6, temp)):
                part[r_id] = 1 - part[r_id]
                part_list = part.tolist()
                for col, _, _ in read_obs[r_id]:
                    b0, b1, _, _, _ = eval_col(col, part_list)
                    h0_cur[col] = b0
                    h1_cur[col] = b1
                cur_cost = evaluate_partition(part)[3]

        improved = True
        while improved:
            improved = False
            for r_id in range(n_reads):
                part[r_id] = 1 - part[r_id]
                _, _, _, cand_cost, _ = evaluate_partition(part)
                if cand_cost + 1e-9 < cur_cost:
                    cur_cost = cand_cost
                    improved = True
                else:
                    part[r_id] = 1 - part[r_id]

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_part = part.copy()

    h0_final, h1_final, conf_final, final_cost, final_flips = evaluate_partition(best_part)
    return h0_final, h1_final, conf_final, final_cost, final_flips


def compute_switch_error(
    pred_h0: Sequence[int],
    pred_h1: Sequence[int],
    true_h0: Sequence[int],
    true_h1: Sequence[int],
) -> Tuple[float, float]:
    """Compute switch error rate and Hamming error rate across heterozygous sites."""
    if len(pred_h0) != len(true_h0) or len(pred_h1) != len(true_h1):
        raise PhasingConvergenceError("Haplotype vectors must have identical lengths")

    het_positions = [i for i in range(len(true_h0)) if true_h0[i] != true_h1[i]]
    if not het_positions:
        return 0.0, 0.0

    err_direct = sum(
        1 for i in het_positions if not (pred_h0[i] == true_h0[i] and pred_h1[i] == true_h1[i])
    )
    err_swapped = sum(
        1 for i in het_positions if not (pred_h0[i] == true_h1[i] and pred_h1[i] == true_h0[i])
    )
    hamming_rate = min(err_direct, err_swapped) / len(het_positions)

    if len(het_positions) < 2:
        return 0.0, hamming_rate

    switches = 0
    prev_orientation: Optional[bool] = None
    for i in het_positions:
        if pred_h0[i] == pred_h1[i]:
            continue
        is_direct = pred_h0[i] == true_h0[i]
        if prev_orientation is not None and is_direct != prev_orientation:
            switches += 1
        prev_orientation = is_direct

    switch_rate = switches / max(1, len(het_positions) - 1)
    return switch_rate, hamming_rate


def compute_block_n50(spans_bp: Sequence[int]) -> int:
    """Compute physical N50 span (bp) across phased blocks."""
    valid = sorted((s for s in spans_bp if s > 0), reverse=True)
    if not valid:
        return 0
    half_total = sum(valid) / 2.0
    acc = 0
    for s in valid:
        acc += s
        if acc >= half_total:
            return s
    return valid[-1]


def phase_locus_reads(
    locus: PharmacogeneLocus,
    fragments: Sequence[ReadFragment],
    exact_dp_max_reads: int = 12,
    truth_haplotypes: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    seed: int = 42,
) -> PhasingResult:
    """Phase all connected variant components in a pharmacogene locus."""
    n_sites = len(locus.variants)
    if n_sites == 0:
        raise PhasingConvergenceError(f"Locus {locus.gene_symbol} contains no variant sites")

    site_zygosities = _infer_site_zygosities(locus, fragments)
    components = extract_phase_components(n_sites, fragments)
    blocks: List[PhasedBlock] = []
    methods_used: set[str] = set()
    phased_sites: set[int] = set()

    for b_id, comp_sites in enumerate(components):
        comp_set = set(comp_sites)
        comp_reads = [
            (idx, frag)
            for idx, frag in enumerate(fragments)
            if len(comp_set.intersection(frag.observations.keys())) >= 2
        ]
        if not comp_reads:
            continue

        site_to_col = {s: c for c, s in enumerate(comp_sites)}
        col_depth = [0] * len(comp_sites)
        for _, frag in comp_reads:
            cols = [site_to_col[s] for s in frag.observations if s in site_to_col]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols) + 1):
                    col_depth[c] += 1
        max_col_depth = max(col_depth) if col_depth else 0

        if max_col_depth <= min(12, exact_dp_max_reads):
            h0, h1, conf, mec_cost, flips = _solve_exact_dp_component(
                comp_sites, comp_reads, site_zygosities
            )
            methods_used.add("exact_bitmask_dp")
        else:
            h0, h1, conf, mec_cost, flips = _solve_annealed_component(
                comp_sites, comp_reads, site_zygosities, seed=seed + b_id
            )
            methods_used.add("simulated_annealing_mec")

        positions = [locus.variants[i].pos for i in comp_sites]
        span_bp = max(positions) - min(positions)
        blocks.append(
            PhasedBlock(
                block_id=b_id,
                variant_indices=comp_sites,
                haplotype_0=h0,
                haplotype_1=h1,
                site_confidence=conf,
                mec_cost=mec_cost,
                unweighted_flips=flips,
                spanning_reads=len(comp_reads),
                genomic_span_bp=max(1, span_bp),
            )
        )
        phased_sites.update(comp_sites)

    total_mec = sum(b.mec_cost for b in blocks)
    total_flips = sum(b.unweighted_flips for b in blocks)
    phased_fraction = len(phased_sites) / n_sites
    n50_bp = compute_block_n50([b.genomic_span_bp for b in blocks])
    method_label = "+".join(sorted(methods_used)) if methods_used else "unphased"

    switch_err: Optional[float] = None
    hamming_err: Optional[float] = None
    if truth_haplotypes is not None and blocks:
        t0, t1 = truth_haplotypes
        largest = max(blocks, key=lambda b: len(b.variant_indices))
        sub_t0 = [t0[i] for i in largest.variant_indices]
        sub_t1 = [t1[i] for i in largest.variant_indices]
        switch_err, hamming_err = compute_switch_error(
            largest.haplotype_0, largest.haplotype_1, sub_t0, sub_t1
        )

    return PhasingResult(
        locus_name=locus.gene_symbol,
        blocks=blocks,
        total_mec_cost=total_mec,
        total_unweighted_flips=total_flips,
        phased_fraction=phased_fraction,
        n50_bp=n50_bp,
        solver_method=method_label,
        switch_error_rate=switch_err,
        hamming_error_rate=hamming_err,
    )
