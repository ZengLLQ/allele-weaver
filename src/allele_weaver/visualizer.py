"""Pure 7-bit ASCII terminal visualizers for pharmacogenomics & PBPK telemetry.

Strictly uses ASCII codes 0-127 (+, -, |, *, #, ., =, :, ^, v) so Windows
cp1252 consoles and headless CI runners never encounter UnicodeEncodeError.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from allele_weaver.models import (
    CNVPosteriorProfile,
    CNVState,
    DiplotypeCall,
    DrugPKParams,
    LDMatrixResult,
    PharmacogeneLocus,
    PhasingResult,
    PKSimulationResult,
)

_CNV_CHAR = {
    CNVState.DEL_0: "0",
    CNVState.HET_DEL_1: "1",
    CNVState.NEUTRAL_2: "=",
    CNVState.DUP_3: "^",
    CNVState.MULTI_4: "#",
    CNVState.HYBRID_CONV: "X",
}


def render_haplotype_tracks(locus: PharmacogeneLocus, phasing: PhasingResult) -> str:
    """Render phased haplotype tracks (H0 / H1) and per-site MEC confidence."""
    lines = [
        f"=== HAPLOTYPE PHASING TRACKS: {locus.gene_symbol} ({phasing.solver_method}) ===",
        f"Blocks: {len(phasing.blocks)} | Phased: {phasing.phased_fraction * 100:.1f}% | "
        f"N50: {phasing.n50_bp} bp | MEC cost: {phasing.total_mec_cost:.2f} "
        f"({phasing.total_unweighted_flips} flips)",
    ]
    if not phasing.blocks:
        lines.append("  [no multi-site phased blocks recovered]")
        return "\n".join(lines)

    for blk in phasing.blocks:
        lines.append(
            f"  Block #{blk.block_id} (sites={len(blk.variant_indices)}, "
            f"reads={blk.spanning_reads}, span={blk.genomic_span_bp} bp):"
        )
        header_idx = "    rsID        : " + " ".join(
            f"{locus.variants[i].rsid[-6:]:>6s}" for i in blk.variant_indices
        )
        track_h0 = "    Haplotype 0 : " + " ".join(f"{b:>6d}" for b in blk.haplotype_0)
        track_h1 = "    Haplotype 1 : " + " ".join(f"{b:>6d}" for b in blk.haplotype_1)
        track_cf = "    Conf (LLR)  : " + " ".join(f"{c:>6.1f}" for c in blk.site_confidence)
        lines.extend([header_idx, track_h0, track_h1, track_cf])
    return "\n".join(lines)


def render_cnv_profile(profile: CNVPosteriorProfile) -> str:
    """Render ASCII copy-number depth ratio chart and Viterbi state ribbon."""
    lines = [
        f"=== CNV HMM PROFILE: {profile.locus_name} "
        f"(Est Copies={profile.estimated_total_copies:.2f}, HybridConv={profile.has_gene_conversion}) ===",
    ]
    n_bins = len(profile.bins)
    ratios = [b.observed_depth / max(1.0, b.expected_diploid_depth) for b in profile.bins]
    max_r = max(2.2, max(ratios) * 1.05) if ratios else 2.2
    height = 7

    for row in range(height, -1, -1):
        level = (row / height) * max_r
        prefix = f"  {level:4.1f}x |"
        chars: List[str] = []
        for r in ratios:
            if abs(r - level) <= (max_r / height) * 0.55:
                chars.append("*")
            elif r >= level and abs(level - 1.0) < 0.15:
                chars.append("-")
            elif r >= level:
                chars.append(":")
            else:
                chars.append(" ")
        lines.append(prefix + " ".join(chars))

    lines.append("        +" + "--" * n_bins)
    state_syms = " ".join(_CNV_CHAR[s] for s in profile.viterbi_path)
    bin_nums = " ".join(f"{i % 10}" for i in range(n_bins))
    lines.append(f"  State | {state_syms}")
    lines.append(f"  Bin   | {bin_nums}")
    lines.append("  Legend: 0=DEL_0  1=HET_DEL_1  ==NEUTRAL_2  ^=DUP_3  #=MULTI_4  X=HYBRID_CONV")
    return "\n".join(lines)


def render_diplotype_posteriors(calls: Sequence[DiplotypeCall], width: int = 28) -> str:
    """Render ranked Bayesian star-allele diplotype posterior bar chart."""
    lines = ["=== STAR-ALLELE DIPLOTYPE POSTERIORS ==="]
    for idx, c in enumerate(calls, start=1):
        bar_len = int(round(c.posterior_prob * width))
        bar = "#" * bar_len + "." * max(0, width - bar_len)
        lines.append(
            f"  {idx}. {c.label:<12s} |{bar}| P={c.posterior_prob:6.3f} | "
            f"AS={c.activity_score:4.2f} | {c.phenotype.value}"
        )
    return "\n".join(lines)


def render_pk_curve(
    sim: PKSimulationResult,
    drug: DrugPKParams,
    cols: int = 56,
    rows: int = 10,
) -> str:
    """Render ASCII PBPK concentration-time curve with therapeutic window markers."""
    t_arr = np.asarray(sim.time_h, dtype=float)
    c_arr = np.asarray(
        sim.c_metabolite_mg_per_l if drug.is_prodrug else sim.c_plasma_mg_per_l,
        dtype=float,
    )
    moiety = "Active Metabolite" if drug.is_prodrug else "Central Plasma"

    y_max = max(drug.c_tox_max_mg_per_l * 1.15, float(np.max(c_arr)) * 1.10, 1e-4)
    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    # Draw therapeutic ceiling (c_tox_max) and floor (c_eff_min)
    tox_r = rows - 1 - int(round(np.clip(drug.c_tox_max_mg_per_l / y_max, 0.0, 1.0) * (rows - 1)))
    eff_r = rows - 1 - int(round(np.clip(drug.c_eff_min_mg_per_l / y_max, 0.0, 1.0) * (rows - 1)))
    for c in range(cols):
        grid[tox_r][c] = "="
        grid[eff_r][c] = "-"

    # Sample trajectory across columns
    t_grid = np.linspace(float(t_arr[0]), float(t_arr[-1]), cols)
    c_sampled = np.interp(t_grid, t_arr, c_arr)
    for col_i, val in enumerate(c_sampled):
        r_i = rows - 1 - int(round(np.clip(val / y_max, 0.0, 1.0) * (rows - 1)))
        grid[r_i][col_i] = "*"

    lines = [
        f"=== PBPK EXPOSURE PROFILE: {drug.drug_name.upper()} ({moiety}) | {sim.diplotype_label} (AS={sim.activity_score:.2f}) ===",
        f"C_max={sim.c_max:.4f} mg/L | C_trough={sim.c_trough:.4f} mg/L | "
        f"AUC={sim.auc_last_interval:.3f} mg*h/L | TTR={sim.time_in_therapeutic_window_frac * 100:.1f}%",
    ]
    for r_i in range(rows):
        y_val = y_max * (1.0 - r_i / max(1, rows - 1))
        tag = " "
        if r_i == tox_r:
            tag = "T"
        elif r_i == eff_r:
            tag = "E"
        lines.append(f"  {y_val:6.3f} {tag}|{''.join(grid[r_i])}|")
    lines.append("         +" + "-" * cols + "+")
    lines.append(
        f"          0.0 h{' ' * max(1, cols - 14)}{t_arr[-1]:5.1f} h   "
        "[=:C_tox, -:C_eff, *:Conc]"
    )
    return "\n".join(lines)


def render_ld_heatmap(ld: LDMatrixResult) -> str:
    """Render lower-triangular 7-bit ASCII Linkage Disequilibrium (r^2) matrix."""
    lines = ["=== LINKAGE DISEQUILIBRIUM (r^2) MATRIX ==="]
    n = len(ld.rsids)
    for i in range(n):
        cells: List[str] = []
        for j in range(i + 1):
            r2 = ld.r_squared[i][j]
            if r2 >= 0.8:
                sym = "#"
            elif r2 >= 0.5:
                sym = "+"
            elif r2 >= 0.2:
                sym = ":"
            else:
                sym = "."
            cells.append(f"{sym}({r2:.2f})")
        lines.append(f"  {ld.rsids[i]:<11s} | " + " ".join(cells))
    lines.append("  Legend: # >=0.80  + >=0.50  : >=0.20  . <0.20")
    return "\n".join(lines)
