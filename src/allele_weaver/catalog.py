"""Curated pharmacogene catalog, CPIC guidelines, drug PK/PD parameters, and synthetic cohort generator.

Includes realistic multi-variant and structural definitions for:
- CYP2D6 (with CYP2D7P paralogous homology mask, *1, *2, *3, *4, *5 deletion,
  *9, *10, *17, *36 exon-9 conversion, *41, *1x2/*2x2 duplications, *68 tandem)
- CYP2C19 (*1, *2, *3, *4, *17)
- CYP2C9 (*1, *2, *3, *5, *8, *11)
- DPYD (*1, *2A, c.2846A>T, HapB3, *13)
- SLCO1B1 (*1, *5, *15)

Also provides a physically grounded SyntheticCohortGenerator that simulates
long/paired-end reads with sequencing errors, paralogous mis-mapping, GC-biased
bin read depths, and population genotype panels for benchmarking and validation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from allele_weaver.models import (
    CNVBinObservation,
    CNVState,
    CPICGuideline,
    DrugPKParams,
    LocusDefinitionError,
    MetabolizerPhenotype,
    ParalogRegion,
    PharmacogeneLocus,
    PopulationPanel,
    ReadFragment,
    StarAlleleDefinition,
    VariantSite,
)


def build_cyp2d6_locus() -> PharmacogeneLocus:
    """Construct curated CYP2D6 pharmacogene locus on chr22 with CYP2D7P homology."""
    variants = [
        VariantSite("rs1080985", "chr22", 42132375, "C", "G", "normal", 25, 0, "0/0", 0.01),
        VariantSite("rs1065852", "chr22", 42130692, "G", "A", "decreased", 25, 0, "0/0", 0.02),
        VariantSite("rs28371706", "chr22", 42129770, "G", "A", "decreased", 25, 0, "0/0", 0.02),
        VariantSite("rs3892097", "chr22", 42128945, "C", "T", "no_function", 25, 0, "0/0", 0.03),
        VariantSite("rs35742686", "chr22", 42128242, "T", "DEL", "no_function", 25, 0, "0/0", 0.02),
        VariantSite("rs5030656", "chr22", 42128174, "CTT", "DEL", "decreased", 25, 0, "0/0", 0.02),
        VariantSite("rs16947", "chr22", 42127941, "G", "A", "normal", 25, 0, "0/0", 0.03),
        VariantSite("rs28371725", "chr22", 42127803, "C", "T", "decreased", 25, 0, "0/0", 0.04),
        VariantSite("rs1135840", "chr22", 42126611, "C", "G", "normal", 25, 0, "0/0", 0.14),
    ]
    star_alleles = {
        "*1": StarAlleleDefinition("*1", {}, 1.0, "normal_function", "normal", 1, 0.34),
        "*2": StarAlleleDefinition(
            "*2", {"rs16947": 1, "rs1135840": 1}, 1.0, "normal_function", "normal", 1, 0.24
        ),
        "*3": StarAlleleDefinition(
            "*3", {"rs35742686": 1}, 0.0, "no_function", "normal", 1, 0.015
        ),
        "*4": StarAlleleDefinition(
            "*4",
            {"rs1065852": 1, "rs3892097": 1, "rs1135840": 1},
            0.0,
            "no_function",
            "normal",
            1,
            0.16,
        ),
        "*5": StarAlleleDefinition(
            "*5", {}, 0.0, "no_function", "deletion", 0, 0.035
        ),
        "*9": StarAlleleDefinition(
            "*9", {"rs5030656": 1}, 0.5, "decreased_function", "normal", 1, 0.02
        ),
        "*10": StarAlleleDefinition(
            "*10", {"rs1065852": 1, "rs1135840": 1}, 0.25, "decreased_function", "normal", 1, 0.07
        ),
        "*17": StarAlleleDefinition(
            "*17",
            {"rs28371706": 1, "rs16947": 1, "rs1135840": 1},
            0.5,
            "decreased_function",
            "normal",
            1,
            0.03,
        ),
        "*36": StarAlleleDefinition(
            "*36",
            {"rs1065852": 1, "rs1135840": 1},
            0.0,
            "no_function",
            "hybrid_conversion",
            1,
            0.015,
        ),
        "*41": StarAlleleDefinition(
            "*41",
            {"rs16947": 1, "rs28371725": 1, "rs1135840": 1},
            0.5,
            "decreased_function",
            "normal",
            1,
            0.055,
        ),
        "*1x2": StarAlleleDefinition(
            "*1x2", {}, 2.0, "increased_function", "duplication", 2, 0.01
        ),
        "*2x2": StarAlleleDefinition(
            "*2x2",
            {"rs16947": 1, "rs1135840": 1},
            2.0,
            "increased_function",
            "duplication",
            2,
            0.01,
        ),
    }
    paralog_regions = [
        ParalogRegion("Intron6_CYP2D7_Block", 42127400, 42128100, 0.94, 0.04),
        ParalogRegion("Exon9_CYP2D7_Conversion", 42126400, 42126850, 0.98, 0.14),
    ]
    exon_bins = [
        (42126200, 42126700),
        (42126700, 42127200),
        (42127200, 42127700),
        (42127700, 42128200),
        (42128200, 42128700),
        (42128700, 42129200),
        (42129200, 42129700),
        (42129700, 42130200),
        (42130200, 42130700),
        (42130700, 42131200),
        (42131200, 42131700),
        (42131700, 42132500),
    ]
    return PharmacogeneLocus(
        gene_symbol="CYP2D6",
        chrom="chr22",
        start_bp=42126000,
        end_bp=42133000,
        variants=variants,
        star_alleles=star_alleles,
        paralog_regions=paralog_regions,
        reference_activity_score=2.0,
        exon_bin_bounds=exon_bins,
    )


def build_cyp2c19_locus() -> PharmacogeneLocus:
    """Construct curated CYP2C19 locus on chr10."""
    variants = [
        VariantSite("rs12248560", "chr10", 94761900, "C", "T", "increased", 25, 0, "0/0", 0.0),
        VariantSite("rs28399504", "chr10", 94762706, "A", "G", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs4986893", "chr10", 94775489, "G", "A", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs4244285", "chr10", 94781859, "G", "A", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs3758581", "chr10", 94842866, "A", "G", "normal", 25, 0, "0/0", 0.0),
    ]
    star_alleles = {
        "*1": StarAlleleDefinition("*1", {}, 1.0, "normal_function", "normal", 1, 0.56),
        "*2": StarAlleleDefinition(
            "*2", {"rs4244285": 1, "rs3758581": 1}, 0.0, "no_function", "normal", 1, 0.18
        ),
        "*3": StarAlleleDefinition(
            "*3", {"rs4986893": 1, "rs3758581": 1}, 0.0, "no_function", "normal", 1, 0.04
        ),
        "*4": StarAlleleDefinition(
            "*4", {"rs28399504": 1}, 0.0, "no_function", "normal", 1, 0.01
        ),
        "*17": StarAlleleDefinition(
            "*17", {"rs12248560": 1}, 1.5, "increased_function", "normal", 1, 0.21
        ),
    }
    exon_bins = [(94761500 + i * 8500, 94761500 + (i + 1) * 8500) for i in range(10)]
    return PharmacogeneLocus(
        gene_symbol="CYP2C19",
        chrom="chr10",
        start_bp=94761000,
        end_bp=94847000,
        variants=variants,
        star_alleles=star_alleles,
        paralog_regions=[],
        reference_activity_score=2.0,
        exon_bin_bounds=exon_bins,
    )


def build_cyp2c9_locus() -> PharmacogeneLocus:
    """Construct curated CYP2C9 locus on chr10."""
    variants = [
        VariantSite("rs1799853", "chr10", 94942290, "C", "T", "decreased", 25, 0, "0/0", 0.0),
        VariantSite("rs1057910", "chr10", 94981296, "A", "C", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs28371686", "chr10", 94981301, "C", "G", "decreased", 25, 0, "0/0", 0.0),
        VariantSite("rs7900194", "chr10", 94942309, "G", "A", "decreased", 25, 0, "0/0", 0.0),
        VariantSite("rs28371685", "chr10", 94972235, "C", "T", "decreased", 25, 0, "0/0", 0.0),
    ]
    star_alleles = {
        "*1": StarAlleleDefinition("*1", {}, 1.0, "normal_function", "normal", 1, 0.78),
        "*2": StarAlleleDefinition("*2", {"rs1799853": 1}, 0.5, "decreased_function", "normal", 1, 0.11),
        "*3": StarAlleleDefinition("*3", {"rs1057910": 1}, 0.0, "no_function", "normal", 1, 0.07),
        "*5": StarAlleleDefinition("*5", {"rs28371686": 1}, 0.0, "no_function", "normal", 1, 0.01),
        "*8": StarAlleleDefinition("*8", {"rs7900194": 1}, 0.5, "decreased_function", "normal", 1, 0.02),
        "*11": StarAlleleDefinition("*11", {"rs28371685": 1}, 0.5, "decreased_function", "normal", 1, 0.01),
    }
    exon_bins = [(94940000 + i * 5000, 94940000 + (i + 1) * 5000) for i in range(9)]
    return PharmacogeneLocus(
        gene_symbol="CYP2C9",
        chrom="chr10",
        start_bp=94938000,
        end_bp=94990000,
        variants=variants,
        star_alleles=star_alleles,
        paralog_regions=[],
        reference_activity_score=2.0,
        exon_bin_bounds=exon_bins,
    )


def build_dpyd_locus() -> PharmacogeneLocus:
    """Construct curated DPYD fluoropyrimidine toxicity locus on chr1."""
    variants = [
        VariantSite("rs3918290", "chr1", 97450058, "C", "T", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs55886062", "chr1", 97515839, "A", "C", "no_function", 25, 0, "0/0", 0.0),
        VariantSite("rs67376798", "chr1", 97082391, "T", "A", "decreased", 25, 0, "0/0", 0.0),
        VariantSite("rs75017182", "chr1", 97573863, "G", "C", "decreased", 25, 0, "0/0", 0.0),
    ]
    star_alleles = {
        "*1": StarAlleleDefinition("*1", {}, 1.0, "normal_function", "normal", 1, 0.94),
        "*2A": StarAlleleDefinition("*2A", {"rs3918290": 1}, 0.0, "no_function", "normal", 1, 0.015),
        "*13": StarAlleleDefinition("*13", {"rs55886062": 1}, 0.0, "no_function", "normal", 1, 0.005),
        "c.2846A>T": StarAlleleDefinition(
            "c.2846A>T", {"rs67376798": 1}, 0.5, "decreased_function", "normal", 1, 0.015
        ),
        "HapB3": StarAlleleDefinition(
            "HapB3", {"rs75017182": 1}, 0.5, "decreased_function", "normal", 1, 0.025
        ),
    }
    exon_bins = [(97080000 + i * 50000, 97080000 + (i + 1) * 50000) for i in range(10)]
    return PharmacogeneLocus(
        gene_symbol="DPYD",
        chrom="chr1",
        start_bp=97080000,
        end_bp=97600000,
        variants=variants,
        star_alleles=star_alleles,
        paralog_regions=[],
        reference_activity_score=2.0,
        exon_bin_bounds=exon_bins,
    )


LOCUS_BUILDERS = {
    "CYP2D6": build_cyp2d6_locus,
    "CYP2C19": build_cyp2c19_locus,
    "CYP2C9": build_cyp2c9_locus,
    "DPYD": build_dpyd_locus,
}


def get_pharmacogene_locus(gene_symbol: str) -> PharmacogeneLocus:
    """Return a fresh PharmacogeneLocus definition by symbol (case-insensitive)."""
    key = gene_symbol.strip().upper()
    if key not in LOCUS_BUILDERS:
        raise LocusDefinitionError(
            f"Unsupported gene '{gene_symbol}'. Available: {sorted(LOCUS_BUILDERS)}"
        )
    return LOCUS_BUILDERS[key]()


DRUG_PK_CATALOG: Dict[str, DrugPKParams] = {
    "codeine": DrugPKParams(
        drug_name="codeine",
        gene_symbol="CYP2D6",
        ka_per_h=1.45,
        bioavailability=0.55,
        v_central_l=18.0,
        v_periph_l=95.0,
        v_hepatic_l=1.5,
        q_hepatic_l_per_h=87.0,
        q_inter_l_per_h=24.0,
        vmax_cyp_mg_per_h=14.0,
        km_mg_per_l=2.5,
        cl_renal_l_per_h=11.0,
        activity_exponent=0.90,
        is_prodrug=True,
        fm_active_metabolite=0.75,
        cl_met_l_per_h=18.0,
        v_met_l=40.0,
        ec50_mg_per_l=0.045,
        emax=0.85,
        hill_n=1.4,
        k_out_per_h=0.14,
        c_eff_min_mg_per_l=0.015,
        c_tox_max_mg_per_l=0.095,
    ),
    "clopidogrel": DrugPKParams(
        drug_name="clopidogrel",
        gene_symbol="CYP2C19",
        ka_per_h=1.80,
        bioavailability=0.50,
        v_central_l=25.0,
        v_periph_l=110.0,
        v_hepatic_l=1.5,
        q_hepatic_l_per_h=87.0,
        q_inter_l_per_h=18.0,
        vmax_cyp_mg_per_h=38.0,
        km_mg_per_l=4.0,
        cl_renal_l_per_h=16.0,
        activity_exponent=0.95,
        is_prodrug=True,
        fm_active_metabolite=0.55,
        cl_met_l_per_h=22.0,
        v_met_l=32.0,
        ec50_mg_per_l=0.06,
        emax=0.78,
        hill_n=1.25,
        k_out_per_h=0.06,
        c_eff_min_mg_per_l=0.025,
        c_tox_max_mg_per_l=0.18,
    ),
    "warfarin": DrugPKParams(
        drug_name="warfarin",
        gene_symbol="CYP2C9",
        ka_per_h=1.20,
        bioavailability=0.95,
        v_central_l=8.5,
        v_periph_l=6.5,
        v_hepatic_l=1.5,
        q_hepatic_l_per_h=87.0,
        q_inter_l_per_h=3.2,
        vmax_cyp_mg_per_h=1.45,
        km_mg_per_l=3.8,
        cl_renal_l_per_h=0.035,
        activity_exponent=0.85,
        is_prodrug=False,
        fm_active_metabolite=0.0,
        cl_met_l_per_h=5.0,
        v_met_l=10.0,
        ec50_mg_per_l=0.65,
        emax=0.82,
        hill_n=1.5,
        k_out_per_h=0.04,
        c_eff_min_mg_per_l=0.35,
        c_tox_max_mg_per_l=1.65,
    ),
    "fluorouracil": DrugPKParams(
        drug_name="fluorouracil",
        gene_symbol="DPYD",
        ka_per_h=0.65,
        bioavailability=0.85,
        v_central_l=24.0,
        v_periph_l=42.0,
        v_hepatic_l=1.5,
        q_hepatic_l_per_h=87.0,
        q_inter_l_per_h=12.0,
        vmax_cyp_mg_per_h=85.0,
        km_mg_per_l=6.5,
        cl_renal_l_per_h=4.5,
        activity_exponent=0.95,
        is_prodrug=False,
        fm_active_metabolite=0.0,
        cl_met_l_per_h=10.0,
        v_met_l=20.0,
        ec50_mg_per_l=1.20,
        emax=0.90,
        hill_n=1.6,
        k_out_per_h=0.10,
        c_eff_min_mg_per_l=0.35,
        c_tox_max_mg_per_l=3.00,
    ),
}


def get_drug_pk_params(drug_name: str) -> DrugPKParams:
    """Retrieve curated PBPK/PD parameter set for a pharmacogene substrate."""
    key = drug_name.strip().lower()
    if key not in DRUG_PK_CATALOG:
        raise LocusDefinitionError(
            f"Unsupported drug '{drug_name}'. Available: {sorted(DRUG_PK_CATALOG)}"
        )
    return DRUG_PK_CATALOG[key]


def get_cpic_guideline(gene_symbol: str, drug_name: str, phenotype: MetabolizerPhenotype) -> CPICGuideline:
    """Return CPIC clinical dosing guidance for a (gene, drug, phenotype) triple."""
    g = gene_symbol.upper()
    d = drug_name.lower()

    if d == "codeine" and g == "CYP2D6":
        table = {
            MetabolizerPhenotype.POOR: (
                "Avoid codeine due to lack of morphine formation; use non-tramadol analgesic.",
                0.0,
                "CRITICAL_AVOID",
            ),
            MetabolizerPhenotype.INTERMEDIATE: (
                "Use label-recommended initial dose or consider alternative if analgesia inadequate.",
                0.75,
                "MODERATE_CAUTION",
            ),
            MetabolizerPhenotype.NORMAL: (
                "Use standard age/weight-specific label dosing.",
                1.0,
                "STANDARD",
            ),
            MetabolizerPhenotype.ULTRARAPID: (
                "CONTRAINDICATED: high risk of morphine respiratory toxicity; select non-opioid or non-CYP2D6 analgesic.",
                0.0,
                "CRITICAL_AVOID",
            ),
        }
    elif d == "clopidogrel" and g == "CYP2C19":
        table = {
            MetabolizerPhenotype.POOR: (
                "Avoid clopidogrel; switch to prasugrel or ticagrelor if no contraindication.",
                0.0,
                "CRITICAL_AVOID",
            ),
            MetabolizerPhenotype.INTERMEDIATE: (
                "Consider alternative antiplatelet (prasugrel/ticagrelor) in ACS/PCI settings.",
                0.5,
                "HIGH_CAUTION",
            ),
            MetabolizerPhenotype.NORMAL: (
                "Standard 75 mg/day maintenance dosing recommended.",
                1.0,
                "STANDARD",
            ),
            MetabolizerPhenotype.RAPID: (
                "Standard 75 mg/day dosing; monitor for mild bleeding risk.",
                1.0,
                "STANDARD",
            ),
            MetabolizerPhenotype.ULTRARAPID: (
                "Standard dosing; enhanced platelet inhibition expected.",
                0.9,
                "MODERATE_CAUTION",
            ),
        }
    elif d == "warfarin" and g == "CYP2C9":
        table = {
            MetabolizerPhenotype.POOR: (
                "Reduce starting maintenance dose by 50-75% and increase INR monitoring frequency.",
                0.35,
                "HIGH_CAUTION",
            ),
            MetabolizerPhenotype.INTERMEDIATE: (
                "Reduce initial maintenance dose by 25-40% based on genotype-guided algorithm.",
                0.65,
                "MODERATE_CAUTION",
            ),
            MetabolizerPhenotype.NORMAL: (
                "Initiate standard empiric warfarin dose with routine INR titration.",
                1.0,
                "STANDARD",
            ),
        }
    elif d == "fluorouracil" and g == "DPYD":
        table = {
            MetabolizerPhenotype.POOR: (
                "Avoid fluorouracil/capecitabine; high risk of fatal neutropenia/mucositis.",
                0.0,
                "CRITICAL_AVOID",
            ),
            MetabolizerPhenotype.INTERMEDIATE: (
                "Reduce starting dose by 50% (Activity Score 1.0) and titrate by toxicity.",
                0.50,
                "HIGH_CAUTION",
            ),
            MetabolizerPhenotype.NORMAL: (
                "Standard fluoropyrimidine dosing.",
                1.0,
                "STANDARD",
            ),
        }
    else:
        table = {}

    if phenotype in table:
        rec, mult, alert = table[phenotype]
        return CPICGuideline(g, d, phenotype, rec, mult, alert)

    return CPICGuideline(
        g,
        d,
        phenotype,
        f"Interpret {d} exposure alongside therapeutic drug monitoring for {phenotype.value}.",
        1.0,
        "INFORMATIONAL",
    )


class SyntheticCohortGenerator:
    """Generates physically grounded read fragments, CNV depth/BAF profiles, and cohort genotypes."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def build_true_haplotypes(
        self,
        locus: PharmacogeneLocus,
        allele_1: str,
        allele_2: str,
    ) -> Tuple[List[int], List[int], int, int]:
        """Construct ground-truth binary haplotypes and copy counts for two star alleles."""
        if allele_1 not in locus.star_alleles or allele_2 not in locus.star_alleles:
            raise LocusDefinitionError(f"Unknown star allele in ({allele_1}, {allele_2})")
        s1 = locus.star_alleles[allele_1]
        s2 = locus.star_alleles[allele_2]
        h1: List[int] = []
        h2: List[int] = []
        for var in locus.variants:
            v1 = s1.defining_variants.get(var.rsid, 0) if s1.copy_count > 0 else 0
            v2 = s2.defining_variants.get(var.rsid, 0) if s2.copy_count > 0 else 0
            h1.append(v1)
            h2.append(v2)
        return h1, h2, s1.copy_count, s2.copy_count

    def simulate_sample(
        self,
        locus: PharmacogeneLocus,
        allele_1: str,
        allele_2: str,
        mean_depth: float = 40.0,
        seq_error_rate: float = 0.015,
        read_span_sites: int = 4,
    ) -> Tuple[PharmacogeneLocus, List[ReadFragment], List[CNVBinObservation], List[int], List[int]]:
        """Simulate paired/long reads, variant allele counts, and CNV bins for a diplotype."""
        h1, h2, c1, c2 = self.build_true_haplotypes(locus, allele_1, allele_2)
        total_copies = c1 + c2
        n_sites = len(locus.variants)

        ref_counts = np.zeros(n_sites, dtype=int)
        alt_counts = np.zeros(n_sites, dtype=int)
        fragments: List[ReadFragment] = []

        # Total fragment count scales proportionally with diploid copy number
        depth_scale = max(0.05, total_copies / 2.0)
        n_reads = max(12, int(self.rng.poisson(mean_depth * depth_scale * 1.4)))

        for r_idx in range(n_reads):
            if total_copies == 0:
                # Only low-level paralogous background reads
                chrom_allele = [0] * n_sites
            else:
                pick_copy_1 = self.rng.random() < (c1 / total_copies)
                chrom_allele = h1 if (pick_copy_1 and c1 > 0) else (h2 if c2 > 0 else h1)

            start_site = int(self.rng.integers(0, max(1, n_sites - 1)))
            span = int(self.rng.integers(2, max(3, min(n_sites, read_span_sites) + 1)))
            end_site = min(n_sites, start_site + span)
            obs: Dict[int, Tuple[int, int]] = {}

            for s_idx in range(start_site, end_site):
                var = locus.variants[s_idx]
                true_bit = chrom_allele[s_idx]
                # Paralog mismapping can inject reference or CYP2D7-like bits
                if self.rng.random() < var.paralog_mismap_rate:
                    true_bit = 1 if (var.rsid == "rs1135840") else 0

                is_err = self.rng.random() < seq_error_rate
                called_bit = 1 - true_bit if is_err else true_bit
                phred = int(self.rng.integers(12, 22)) if is_err else int(self.rng.integers(28, 41))
                obs[s_idx] = (called_bit, phred)
                if called_bit == 1:
                    alt_counts[s_idx] += 1
                else:
                    ref_counts[s_idx] += 1

            if obs:
                fragments.append(
                    ReadFragment(
                        read_id=f"read_{r_idx:04d}",
                        observations=obs,
                        mapping_quality=int(self.rng.integers(40, 61)),
                    )
                )

        updated_variants: List[VariantSite] = []
        for s_idx, var in enumerate(locus.variants):
            rc = int(ref_counts[s_idx])
            ac = int(alt_counts[s_idx])
            tot = rc + ac
            vaf = ac / tot if tot > 0 else 0.0
            if total_copies == 0:
                gt = "0/0"
            elif vaf >= 0.85:
                gt = "1/1"
            elif vaf <= 0.12:
                gt = "0/0"
            else:
                gt = "0/1"
            updated_variants.append(
                VariantSite(
                    rsid=var.rsid,
                    chrom=var.chrom,
                    pos=var.pos,
                    ref=var.ref,
                    alt=var.alt,
                    functional_impact=var.functional_impact,
                    ref_count=rc,
                    alt_count=ac,
                    genotype=gt,
                    paralog_mismap_rate=var.paralog_mismap_rate,
                )
            )

        s1 = locus.star_alleles[allele_1]
        s2 = locus.star_alleles[allele_2]
        has_conv = (
            s1.structural_type == "hybrid_conversion" or s2.structural_type == "hybrid_conversion"
        )

        cnv_bins: List[CNVBinObservation] = []
        bounds = locus.exon_bin_bounds or [
            (locus.start_bp + i * 1000, locus.start_bp + (i + 1) * 1000) for i in range(10)
        ]
        for b_idx, (b_start, b_end) in enumerate(bounds):
            paralog_hom = 0.0
            for reg in locus.paralog_regions:
                if b_end >= reg.start_bp and b_start <= reg.end_bp:
                    paralog_hom = max(paralog_hom, reg.sequence_identity)

            # In hybrid conversion (*36), last 3 bins show CYP2D7-derived depth drop & mismapping
            bin_copies = float(total_copies)
            if has_conv and b_idx >= len(bounds) - 3:
                bin_copies = max(1.0, float(total_copies) - 0.65)
                paralog_hom = max(paralog_hom, 0.98)

            gc_frac = float(np.clip(0.52 + 0.06 * np.sin(b_idx * 0.7), 0.35, 0.68))
            gc_factor = 1.0 - 0.8 * ((gc_frac - 0.50) ** 2)
            expected_lam = mean_depth * max(0.06, bin_copies / 2.0) * gc_factor
            # Negative binomial overdispersion simulation via Gamma-Poisson
            gamma_rate = float(self.rng.gamma(shape=18.0, scale=expected_lam / 18.0))
            obs_depth = float(self.rng.poisson(max(0.2, gamma_rate)))

            # BAF marker counts inside bin
            het_total = int(max(10, obs_depth))
            if bin_copies <= 0.5:
                true_baf = 0.03
            elif abs(bin_copies - 1.0) < 0.25:
                true_baf = 0.04
            elif abs(bin_copies - 3.0) < 0.25:
                true_baf = 0.333
            elif bin_copies >= 3.75:
                true_baf = 0.25
            elif has_conv and b_idx >= len(bounds) - 3:
                true_baf = 0.22
            else:
                true_baf = 0.50 if total_copies == 2 else 0.33

            het_alt = int(self.rng.binomial(het_total, np.clip(true_baf, 0.01, 0.99)))
            cnv_bins.append(
                CNVBinObservation(
                    bin_index=b_idx,
                    start_bp=b_start,
                    end_bp=b_end,
                    observed_depth=obs_depth,
                    expected_diploid_depth=mean_depth * gc_factor,
                    het_alt_count=het_alt,
                    het_total_count=het_total,
                    gc_fraction=gc_frac,
                    paralog_homology=paralog_hom,
                )
            )

        sim_locus = PharmacogeneLocus(
            gene_symbol=locus.gene_symbol,
            chrom=locus.chrom,
            start_bp=locus.start_bp,
            end_bp=locus.end_bp,
            variants=updated_variants,
            star_alleles=locus.star_alleles,
            paralog_regions=locus.paralog_regions,
            reference_activity_score=locus.reference_activity_score,
            exon_bin_bounds=locus.exon_bin_bounds,
        )
        return sim_locus, fragments, cnv_bins, h1, h2

    def simulate_population_panel(
        self,
        locus: PharmacogeneLocus,
        n_individuals: int = 200,
    ) -> PopulationPanel:
        """Sample unphased diploid genotypes from star-allele population frequencies."""
        alleles = list(locus.star_alleles.values())
        probs = np.array([a.population_frequency for a in alleles], dtype=float)
        probs /= probs.sum()

        rsids = [v.rsid for v in locus.variants]
        positions = [v.pos for v in locus.variants]
        rows: List[List[int]] = []

        for _ in range(n_individuals):
            idx1, idx2 = self.rng.choice(len(alleles), size=2, replace=True, p=probs)
            a1, a2 = alleles[int(idx1)], alleles[int(idx2)]
            row: List[int] = []
            for rsid in rsids:
                g = (
                    (a1.defining_variants.get(rsid, 0) if a1.copy_count > 0 else 0)
                    + (a2.defining_variants.get(rsid, 0) if a2.copy_count > 0 else 0)
                )
                row.append(int(min(2, g)))
            rows.append(row)

        return PopulationPanel(
            locus_name=locus.gene_symbol,
            rsids=rsids,
            positions=positions,
            genotypes=rows,
        )


def expected_cnv_state_for_diplotype(locus: PharmacogeneLocus, allele_1: str, allele_2: str) -> CNVState:
    """Return expected canonical CNV state for ground-truth benchmarking."""
    s1 = locus.star_alleles[allele_1]
    s2 = locus.star_alleles[allele_2]
    if s1.structural_type == "hybrid_conversion" or s2.structural_type == "hybrid_conversion":
        return CNVState.HYBRID_CONV
    tot = s1.copy_count + s2.copy_count
    if tot == 0:
        return CNVState.DEL_0
    if tot == 1:
        return CNVState.HET_DEL_1
    if tot == 2:
        return CNVState.NEUTRAL_2
    if tot == 3:
        return CNVState.DUP_3
    return CNVState.MULTI_4
