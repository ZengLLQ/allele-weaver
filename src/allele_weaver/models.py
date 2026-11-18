"""Core data models, enumerations, and domain invariants for AlleleWeaver.

All physical, genomic, and pharmacokinetic quantities document their canonical
units and validate domain constraints upon instantiation to fail fast on
malformed biological or clinical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AlleleWeaverError(Exception):
    """Base exception for all AlleleWeaver domain and numerical errors."""


class LocusDefinitionError(AlleleWeaverError):
    """Raised when a pharmacogene locus or star-allele table is inconsistent."""


class PhasingConvergenceError(AlleleWeaverError):
    """Raised when MEC phasing encounters invalid or degenerate read matrices."""


class NumericalInstabilityError(AlleleWeaverError):
    """Raised when an HMM, EM, or ODE solver violates numerical stability bounds."""


class CNVState(str, Enum):
    """Discrete structural copy-number states modeled across pharmacogene bins."""

    DEL_0 = "DEL_0"
    HET_DEL_1 = "HET_DEL_1"
    NEUTRAL_2 = "NEUTRAL_2"
    DUP_3 = "DUP_3"
    MULTI_4 = "MULTI_4"
    HYBRID_CONV = "HYBRID_CONV"

    @property
    def total_copies(self) -> int:
        """Return effective diploid copy count in the non-paralog target segment."""
        mapping = {
            CNVState.DEL_0: 0,
            CNVState.HET_DEL_1: 1,
            CNVState.NEUTRAL_2: 2,
            CNVState.DUP_3: 3,
            CNVState.MULTI_4: 4,
            CNVState.HYBRID_CONV: 2,
        }
        return mapping[self]


class MetabolizerPhenotype(str, Enum):
    """CPIC standardized clinical metabolizer phenotype classifications."""

    POOR = "Poor Metabolizer (PM)"
    INTERMEDIATE = "Intermediate Metabolizer (IM)"
    NORMAL = "Normal Metabolizer (NM)"
    RAPID = "Rapid Metabolizer (RM)"
    ULTRARAPID = "Ultrarapid Metabolizer (UM)"
    INDETERMINATE = "Indeterminate"


@dataclass(frozen=True)
class VariantSite:
    """Genomic variant site within a pharmacogene locus.

    Attributes:
        rsid: Variant identifier (e.g. 'rs3892097' or HGVS key).
        chrom: Chromosome identifier without or with 'chr' prefix.
        pos: 1-based genomic coordinate (bp).
        ref: Reference allele nucleotide string.
        alt: Alternate allele nucleotide string.
        functional_impact: Curated impact label ('no_function', 'decreased',
            'normal', 'increased', or 'splice_defect').
        ref_count: Observed reference read depth at this locus.
        alt_count: Observed alternate read depth at this locus.
        genotype: Unphased or phased diploid genotype string ('0/1', '0|1', etc.).
        paralog_mismap_rate: Fraction [0, 0.5) of reads originating from a
            homologous pseudogene (e.g. CYP2D7P) mis-aligned to this site.
    """

    rsid: str
    chrom: str
    pos: int
    ref: str
    alt: str
    functional_impact: str = "normal"
    ref_count: int = 20
    alt_count: int = 0
    genotype: str = "0/0"
    paralog_mismap_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.pos <= 0:
            raise LocusDefinitionError(f"Genomic position must be positive, got {self.pos}")
        if not self.ref or not self.alt:
            raise LocusDefinitionError("Reference and alternate alleles must be non-empty")
        if self.ref_count < 0 or self.alt_count < 0:
            raise LocusDefinitionError("Allele depths must be non-negative")
        if not (0.0 <= self.paralog_mismap_rate < 0.5):
            raise LocusDefinitionError(
                f"paralog_mismap_rate must be in [0, 0.5), got {self.paralog_mismap_rate}"
            )

    @property
    def total_depth(self) -> int:
        """Total read support across reference and alternate alleles."""
        return self.ref_count + self.alt_count

    @property
    def alt_allele_fraction(self) -> float:
        """Observed alternate allele fraction (VAF), returning 0.0 at zero depth."""
        tot = self.total_depth
        return self.alt_count / tot if tot > 0 else 0.0


@dataclass(frozen=True)
class ReadFragment:
    """Sequencing read (or paired-end fragment) spanning variant indices.

    Attributes:
        read_id: Unique read identifier.
        observations: Mapping from 0-based variant index to observed allele
            (0 = reference, 1 = alternate) and Phred base quality (int >= 2).
        mapping_quality: Phred mapping quality (MAPQ).
    """

    read_id: str
    observations: Dict[int, Tuple[int, int]]
    mapping_quality: int = 60

    def __post_init__(self) -> None:
        if self.mapping_quality < 0:
            raise PhasingConvergenceError(f"Negative MAPQ: {self.mapping_quality}")
        for idx, (allele, qual) in self.observations.items():
            if idx < 0:
                raise PhasingConvergenceError(f"Negative variant index: {idx}")
            if allele not in (0, 1):
                raise PhasingConvergenceError(f"Bi-allelic observation must be 0 or 1, got {allele}")
            if qual < 1:
                raise PhasingConvergenceError(f"Phred quality must be >= 1, got {qual}")

    @property
    def span(self) -> Tuple[int, int]:
        """Return inclusive (min_idx, max_idx) variant span covered by read."""
        if not self.observations:
            return (0, -1)
        keys = self.observations.keys()
        return (min(keys), max(keys))


@dataclass
class PhasedBlock:
    """Contiguous phased haplotype block recovered by MEC optimization.

    Attributes:
        block_id: Sequential 0-based component identifier.
        variant_indices: Ordered 0-based indices into the locus variant list.
        haplotype_0: Binary allele vector (0=REF, 1=ALT) for maternal/paternal copy 0.
        haplotype_1: Binary allele vector (0=REF, 1=ALT) for maternal/paternal copy 1.
        site_confidence: Per-site log-odds confidence of phase assignment.
        mec_cost: Weighted Minimum Error Correction flip cost within block.
        unweighted_flips: Raw integer count of discordant allele calls flipped.
        spanning_reads: Number of reads overlapping >= 2 sites in this block.
        genomic_span_bp: Physical span in base pairs between first and last site.
    """

    block_id: int
    variant_indices: List[int]
    haplotype_0: List[int]
    haplotype_1: List[int]
    site_confidence: List[float]
    mec_cost: float
    unweighted_flips: int
    spanning_reads: int
    genomic_span_bp: int


@dataclass
class PhasingResult:
    """Complete output of read-backed MEC haplotype phasing for a locus."""

    locus_name: str
    blocks: List[PhasedBlock]
    total_mec_cost: float
    total_unweighted_flips: int
    phased_fraction: float
    n50_bp: int
    solver_method: str
    switch_error_rate: Optional[float] = None
    hamming_error_rate: Optional[float] = None


@dataclass(frozen=True)
class ParalogRegion:
    """Genomic interval exhibiting high sequence homology with a pseudogene.

    Attributes:
        name: Region label (e.g. 'Exon9_CYP2D7_Homology').
        start_bp: Start genomic position (1-based inclusive).
        end_bp: End genomic position (1-based inclusive).
        sequence_identity: Fractional identity in (0, 1].
        expected_mismap_fraction: Expected fraction of paralogous reads leaking
            into the functional gene bin.
    """

    name: str
    start_bp: int
    end_bp: int
    sequence_identity: float
    expected_mismap_fraction: float


@dataclass(frozen=True)
class StarAlleleDefinition:
    """Definition of a single pharmacogene star allele (*1, *2, *4, *36, etc.).

    Attributes:
        name: Canonical allele designation (e.g. '*4', '*1x2', '*36').
        defining_variants: Mapping from rsid to required allele (1=ALT, 0=REF).
        activity_value: CPIC numeric activity score contribution (e.g. 0.0, 0.5, 1.0, 2.0).
        function_status: Functional classification string.
        structural_type: Structural class ('normal', 'deletion', 'duplication', 'hybrid_conversion').
        copy_count: Number of functional gene copies represented on one chromosome.
        population_frequency: Prior population allele frequency in (0, 1].
    """

    name: str
    defining_variants: Dict[str, int]
    activity_value: float
    function_status: str
    structural_type: str = "normal"
    copy_count: int = 1
    population_frequency: float = 0.05

    def __post_init__(self) -> None:
        if self.activity_value < 0.0:
            raise LocusDefinitionError(f"Activity value cannot be negative: {self.activity_value}")
        if self.copy_count < 0:
            raise LocusDefinitionError(f"Copy count cannot be negative: {self.copy_count}")
        if not (0.0 < self.population_frequency <= 1.0):
            raise LocusDefinitionError(
                f"population_frequency must be in (0, 1], got {self.population_frequency}"
            )


@dataclass
class PharmacogeneLocus:
    """Complete pharmacogene reference definition for phasing, CNV, and star calling."""

    gene_symbol: str
    chrom: str
    start_bp: int
    end_bp: int
    variants: List[VariantSite]
    star_alleles: Dict[str, StarAlleleDefinition]
    paralog_regions: List[ParalogRegion] = field(default_factory=list)
    reference_activity_score: float = 2.0
    exon_bin_bounds: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.end_bp <= self.start_bp:
            raise LocusDefinitionError("Locus end_bp must exceed start_bp")
        if "*1" not in self.star_alleles:
            raise LocusDefinitionError(f"Locus {self.gene_symbol} must define reference '*1' allele")
        if self.reference_activity_score <= 0.0:
            raise LocusDefinitionError("reference_activity_score must be positive")


@dataclass(frozen=True)
class CNVBinObservation:
    """Read-depth and B-allele-frequency observation for one genomic bin.

    Attributes:
        bin_index: 0-based sequential bin index.
        start_bp: Bin start genomic coordinate.
        end_bp: Bin end genomic coordinate.
        observed_depth: Raw or GC-adjusted fragment depth in this bin.
        expected_diploid_depth: Baseline diploid depth calibrated for library size.
        het_alt_count: Alternate read count summed at heterozygous/paralog markers.
        het_total_count: Total read count at heterozygous/paralog markers.
        gc_fraction: Local GC content in [0, 1].
        paralog_homology: Fractional paralog sequence identity in [0, 1].
    """

    bin_index: int
    start_bp: int
    end_bp: int
    observed_depth: float
    expected_diploid_depth: float
    het_alt_count: int = 0
    het_total_count: int = 0
    gc_fraction: float = 0.50
    paralog_homology: float = 0.0


@dataclass(frozen=True)
class CNVEvent:
    """Decoded contiguous structural CNV or gene-conversion event."""

    state: CNVState
    start_bin: int
    end_bin: int
    start_bp: int
    end_bp: int
    mean_depth_ratio: float
    mean_posterior_prob: float


@dataclass
class CNVPosteriorProfile:
    """Full Viterbi & Forward-Backward output across locus genomic bins."""

    locus_name: str
    bins: List[CNVBinObservation]
    states: List[CNVState]
    viterbi_path: List[CNVState]
    posterior_matrix: List[List[float]]
    log_likelihood: float
    events: List[CNVEvent]
    estimated_total_copies: float
    has_gene_conversion: bool


@dataclass(frozen=True)
class DiplotypeCall:
    """Ranked candidate diplotype call with Bayesian posterior & CPIC phenotype."""

    allele_1: str
    allele_2: str
    posterior_prob: float
    log_likelihood: float
    log_prior: float
    activity_score: float
    phenotype: MetabolizerPhenotype
    phase_consistent: bool
    cnv_consistent: bool
    supporting_notes: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Canonical sorted diplotype string such as '*1/*4'."""
        pair = sorted((self.allele_1, self.allele_2))
        return f"{pair[0]}/{pair[1]}"


@dataclass(frozen=True)
class CPICGuideline:
    """Clinical Pharmacogenetics Implementation Consortium dosing recommendation."""

    gene_symbol: str
    drug_name: str
    phenotype: MetabolizerPhenotype
    recommendation: str
    dose_multiplier: float
    therapeutic_alert_level: str


@dataclass(frozen=True)
class DrugPKParams:
    """Multi-compartment PBPK/PD kinetic parameters for a pharmacogene substrate.

    Units:
        ka_per_h: First-order oral absorption rate constant (1/h).
        bioavailability: Oral fraction absorbed (dimensionless in (0, 1]).
        v_central_l: Central plasma apparent distribution volume (L).
        v_periph_l: Peripheral tissue compartment volume (L).
        v_hepatic_l: Hepatic compartment volume (L).
        q_hepatic_l_per_h: Hepatic plasma blood flow (L/h).
        q_inter_l_per_h: Inter-compartmental distributional clearance (L/h).
        vmax_cyp_mg_per_h: Reference maximal CYP metabolic velocity at AS_ref (mg/h).
        km_mg_per_l: Michaelis-Menten half-saturation constant (mg/L).
        cl_renal_l_per_h: Non-CYP renal/biliary clearance (L/h).
        activity_exponent: Allometric exponent gamma scaling CYP Vmax with AS.
        is_prodrug: True if therapeutic effect is driven by active metabolite.
        fm_active_metabolite: Fraction of CYP flux forming active metabolite.
        cl_met_l_per_h: Clearance of active metabolite (L/h).
        v_met_l: Distribution volume of active metabolite (L).
        ec50_mg_per_l: PD concentration at 50% maximal target modulation (mg/L).
        emax: Maximal fractional PD response (e.g. INR elevation or inhibition).
        hill_n: Hill sigmoidicity coefficient for PD effect.
        k_out_per_h: Indirect-response PD turnover rate constant (1/h).
        c_eff_min_mg_per_l: Lower therapeutic trough/exposure bound (mg/L).
        c_tox_max_mg_per_l: Upper safety toxicity ceiling (mg/L).
    """

    drug_name: str
    gene_symbol: str
    ka_per_h: float
    bioavailability: float
    v_central_l: float
    v_periph_l: float
    v_hepatic_l: float
    q_hepatic_l_per_h: float
    q_inter_l_per_h: float
    vmax_cyp_mg_per_h: float
    km_mg_per_l: float
    cl_renal_l_per_h: float
    activity_exponent: float = 0.85
    is_prodrug: bool = False
    fm_active_metabolite: float = 0.0
    cl_met_l_per_h: float = 12.0
    v_met_l: float = 35.0
    ec50_mg_per_l: float = 0.5
    emax: float = 0.80
    hill_n: float = 1.3
    k_out_per_h: float = 0.08
    c_eff_min_mg_per_l: float = 0.15
    c_tox_max_mg_per_l: float = 1.80

    def __post_init__(self) -> None:
        positive_fields = (
            self.ka_per_h,
            self.bioavailability,
            self.v_central_l,
            self.v_periph_l,
            self.v_hepatic_l,
            self.q_hepatic_l_per_h,
            self.km_mg_per_l,
            self.v_met_l,
            self.ec50_mg_per_l,
            self.k_out_per_h,
        )
        if any(val <= 0.0 for val in positive_fields):
            raise NumericalInstabilityError(f"PK volume/rate constants must be > 0 for {self.drug_name}")
        if self.c_tox_max_mg_per_l <= self.c_eff_min_mg_per_l:
            raise NumericalInstabilityError("c_tox_max_mg_per_l must exceed c_eff_min_mg_per_l")


@dataclass(frozen=True)
class DosingRegimen:
    """Scheduled oral dosing regimen."""

    dose_mg: float
    interval_h: float
    n_doses: int
    loading_dose_mg: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dose_mg <= 0.0 or self.interval_h <= 0.0 or self.n_doses < 1:
            raise NumericalInstabilityError("Dosing regimen requires positive dose, interval, and n_doses >= 1")


@dataclass
class PKSimulationResult:
    """Time-course telemetry and summary exposure metrics from RKF45 PBPK/PD."""

    drug_name: str
    diplotype_label: str
    activity_score: float
    time_h: List[float]
    c_plasma_mg_per_l: List[float]
    c_hepatic_mg_per_l: List[float]
    c_periph_mg_per_l: List[float]
    c_metabolite_mg_per_l: List[float]
    pd_effect: List[float]
    c_max: float
    c_trough: float
    auc_last_interval: float
    time_in_therapeutic_window_frac: float
    max_mass_balance_error_mg: float
    rkf_steps_taken: int
    rkf_steps_rejected: int


@dataclass
class DoseOptimizationResult:
    """Result of constrained genotype-guided dose optimization."""

    drug_name: str
    diplotype_label: str
    activity_score: float
    optimal_dose_mg: float
    optimal_interval_h: float
    predicted_c_max: float
    predicted_c_trough: float
    predicted_ttr: float
    safety_margin_ratio: float
    feasible: bool
    simulation: PKSimulationResult


@dataclass
class PopulationPanel:
    """Multi-locus unphased diploid genotypes for a cohort of individuals."""

    locus_name: str
    rsids: List[str]
    positions: List[int]
    genotypes: List[List[int]]  # shape (n_samples, n_sites) values in {0, 1, 2, -1}

    def __post_init__(self) -> None:
        n_sites = len(self.rsids)
        if len(self.positions) != n_sites:
            raise LocusDefinitionError("rsids and positions lengths must match")
        for row in self.genotypes:
            if len(row) != n_sites:
                raise LocusDefinitionError("Each individual genotype row must match n_sites")


@dataclass
class LDMatrixResult:
    """Pairwise linkage disequilibrium summary across locus variants."""

    rsids: List[str]
    d_prime: List[List[float]]
    r_squared: List[List[float]]
    hwe_p_values: List[float]
    em_haplotypes: List[Tuple[str, float]]
    em_log_likelihood: float
    em_iterations: int


@dataclass
class UncertaintySummary:
    """Monte Carlo posterior propagation from genomic uncertainty to PK exposure."""

    drug_name: str
    n_draws: int
    auc_mean: float
    auc_p05: float
    auc_p50: float
    auc_p95: float
    c_max_p95: float
    prob_toxicity: float
    prob_subtherapeutic: float
    sensitivity_indices: Dict[str, float]


def phred_to_error_prob(phred_q: int) -> float:
    """Convert Phred base quality score Q to error probability 10^(-Q/10).

    Clamped at [1e-6, 0.5] to prevent log-zero singularities or inverted likelihoods.
    """
    prob = 10.0 ** (-float(phred_q) / 10.0)
    return float(max(1e-6, min(0.4999, prob)))
