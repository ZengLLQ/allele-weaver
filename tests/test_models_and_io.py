"""Unit tests for data models, domain validation invariants, and bioinformatics I/O."""

from __future__ import annotations

import pytest

from allele_weaver.catalog import get_drug_pk_params, get_pharmacogene_locus
from allele_weaver.io_formats import (
    cigar_reference_span,
    export_clinical_json,
    export_pk_telemetry_csv,
    parse_cigar,
    parse_fragment_tsv,
    parse_vcf,
    phred_char_to_q,
    serialize_fragments_tsv,
    write_vcf,
)
from allele_weaver.models import (
    AlleleWeaverError,
    DosingRegimen,
    LocusDefinitionError,
    NumericalInstabilityError,
    PhasingConvergenceError,
    ReadFragment,
    StarAlleleDefinition,
    VariantSite,
    phred_to_error_prob,
)
from allele_weaver.pbpk_dosing import simulate_pbpk_regimen


def test_variant_site_validation_guards() -> None:
    with pytest.raises(LocusDefinitionError):
        VariantSite("rs1", "chr22", 0, "A", "G")
    with pytest.raises(LocusDefinitionError):
        VariantSite("rs1", "chr22", 100, "", "G")
    with pytest.raises(LocusDefinitionError):
        VariantSite("rs1", "chr22", 100, "A", "G", ref_count=-1)
    with pytest.raises(LocusDefinitionError):
        VariantSite("rs1", "chr22", 100, "A", "G", paralog_mismap_rate=0.6)


def test_read_fragment_validation_and_span() -> None:
    frag = ReadFragment("r1", {2: (0, 30), 5: (1, 35)}, mapping_quality=60)
    assert frag.span == (2, 5)
    with pytest.raises(PhasingConvergenceError):
        ReadFragment("r_bad", {0: (2, 30)})
    with pytest.raises(PhasingConvergenceError):
        ReadFragment("r_bad", {-1: (0, 30)})
    with pytest.raises(PhasingConvergenceError):
        ReadFragment("r_bad", {0: (1, 0)})


def test_star_allele_and_phred_conversion() -> None:
    with pytest.raises(LocusDefinitionError):
        StarAlleleDefinition("*x", {}, -0.5, "bad")
    with pytest.raises(LocusDefinitionError):
        StarAlleleDefinition("*x", {}, 1.0, "ok", population_frequency=0.0)
    assert phred_to_error_prob(30) == pytest.approx(0.001, rel=1e-6)
    assert phred_to_error_prob(10) == pytest.approx(0.1, rel=1e-6)
    assert phred_char_to_q("I") == 40
    with pytest.raises(AlleleWeaverError):
        phred_char_to_q(" ")


def test_cigar_parsing_and_reference_span() -> None:
    ops = parse_cigar("50M2I3D25M")
    assert ops == [(50, "M"), (2, "I"), (3, "D"), (25, "M")]
    assert cigar_reference_span("50M2I3D25M") == 78
    with pytest.raises(AlleleWeaverError):
        parse_cigar("50Z")
    with pytest.raises(AlleleWeaverError):
        parse_cigar("*")


def test_vcf_and_fragment_roundtrip() -> None:
    locus = get_pharmacogene_locus("CYP2C19")
    vcf_text = write_vcf(locus.variants, sample_id="CASE_99")
    parsed_sites = parse_vcf(vcf_text)
    assert len(parsed_sites) == len(locus.variants)
    assert parsed_sites[0].rsid == locus.variants[0].rsid
    assert parsed_sites[0].total_depth == locus.variants[0].total_depth

    frags = [
        ReadFragment("r_a", {0: (0, 35), 2: (1, 32)}, mapping_quality=55),
        ReadFragment("r_b", {1: (1, 28), 3: (0, 40)}, mapping_quality=60),
    ]
    tsv = serialize_fragments_tsv(frags)
    reloaded = parse_fragment_tsv(tsv)
    assert reloaded == frags


def test_pk_telemetry_csv_and_json_export(tmp_path) -> None:
    drug = get_drug_pk_params("warfarin")
    reg = DosingRegimen(dose_mg=5.0, interval_h=24.0, n_doses=2)
    sim = simulate_pbpk_regimen(drug, reg, activity_score=2.0)

    csv_path = tmp_path / "pk.csv"
    export_pk_telemetry_csv(sim, csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("time_h,c_plasma_mg_per_l")
    assert len(lines) == len(sim.time_h) + 1

    json_path = tmp_path / "rep.json"
    out = export_clinical_json({"gene": "CYP2C9", "auc": sim.auc_last_interval}, json_path)
    assert '"gene": "CYP2C9"' in out
    assert json_path.exists()

    with pytest.raises(NumericalInstabilityError):
        DosingRegimen(dose_mg=-5.0, interval_h=24.0, n_doses=2)
