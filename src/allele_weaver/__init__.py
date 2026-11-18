"""AlleleWeaver: Clinical pharmacogenomics & precision dosing simulation toolkit.

Provides a coupled computational bioinformatics pipeline integrating:
1. Read-backed haplotype phasing via exact dynamic-programming and
   simulated-annealing Minimum Error Correction (MEC) optimization.
2. Paralog-aware structural copy-number variation (CNV) and gene-conversion
   Hidden Markov Model (HMM) decoding via Viterbi and Forward-Backward.
3. Bayesian combinatorial star-allele diplotype inference and CPIC activity
   score / metabolizer phenotype assignment.
4. Multi-compartment pharmacogene-scaled PBPK/PD simulation using an adaptive
   Runge-Kutta-Fehlberg (RKF45) ODE solver and constrained dose optimization.
5. Population pharmacogenetics (exact HWE, LD matrices, EM haplotype frequency
   estimation) and Monte Carlo / sensitivity uncertainty quantification.
6. Pure 7-bit ASCII terminal visualization and structured telemetry export.
"""

from __future__ import annotations

from allele_weaver.models import (
    AlleleWeaverError,
    CNVBinObservation,
    CNVEvent,
    CNVPosteriorProfile,
    CNVState,
    CPICGuideline,
    DiplotypeCall,
    DoseOptimizationResult,
    DosingRegimen,
    DrugPKParams,
    LDMatrixResult,
    LocusDefinitionError,
    MetabolizerPhenotype,
    NumericalInstabilityError,
    ParalogRegion,
    PhasedBlock,
    PhasingConvergenceError,
    PhasingResult,
    PharmacogeneLocus,
    PKSimulationResult,
    PopulationPanel,
    ReadFragment,
    StarAlleleDefinition,
    UncertaintySummary,
    VariantSite,
)
from allele_weaver.pipeline import ClinicalReport, PharmacogenomicsPipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlleleWeaverError",
    "LocusDefinitionError",
    "PhasingConvergenceError",
    "NumericalInstabilityError",
    "VariantSite",
    "ReadFragment",
    "PhasedBlock",
    "PhasingResult",
    "StarAlleleDefinition",
    "ParalogRegion",
    "PharmacogeneLocus",
    "CNVState",
    "CNVBinObservation",
    "CNVEvent",
    "CNVPosteriorProfile",
    "MetabolizerPhenotype",
    "DiplotypeCall",
    "CPICGuideline",
    "DrugPKParams",
    "DosingRegimen",
    "PKSimulationResult",
    "DoseOptimizationResult",
    "PopulationPanel",
    "LDMatrixResult",
    "UncertaintySummary",
    "ClinicalReport",
    "PharmacogenomicsPipeline",
]
