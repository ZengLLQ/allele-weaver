"""Integration tests for end-to-end pipeline, 7-bit ASCII safety, and all CLI subcommands."""

from __future__ import annotations

import json

from allele_weaver.catalog import SyntheticCohortGenerator, get_drug_pk_params
from allele_weaver.cli import main as cli_main
from allele_weaver.pipeline import PharmacogenomicsPipeline, run_benchmark_suite
from allele_weaver.popgen_uq import analyze_population_ld
from allele_weaver.visualizer import render_ld_heatmap, render_pk_curve


def test_end_to_end_pipeline_and_strict_ascii(cyp2c9_locus, generator) -> None:
    sim_locus, frags, bins, h1, h2 = generator.simulate_sample(
        cyp2c9_locus, "*1", "*3", mean_depth=50.0
    )
    pipe = PharmacogenomicsPipeline(uq_draws=25, seed=123)
    report = pipe.run(
        sim_locus,
        frags,
        bins,
        drug_name="warfarin",
        nominal_dose_mg=5.0,
        truth_haplotypes=(h1, h2),
    )

    assert report.top_call.label == "*1/*3"
    assert report.dose_optimization.feasible is True
    payload = report.to_dict()
    assert payload["top_diplotype"] == "*1/*3"
    assert payload["dose_optimization"]["mass_balance_error_mg"] < 1e-3

    # Verify strict 7-bit ASCII encoding (never crashes Windows cp1252)
    encoded = report.ascii_dashboard.encode("ascii")
    assert len(encoded) > 200


def test_visualizers_strict_ascii_encoding(cyp2d6_locus) -> None:
    gen = SyntheticCohortGenerator(seed=5)
    panel = gen.simulate_population_panel(cyp2d6_locus, n_individuals=60)
    ld = analyze_population_ld(panel)
    hm = render_ld_heatmap(ld)
    hm.encode("ascii")

    drug = get_drug_pk_params("codeine")
    from allele_weaver.models import DosingRegimen
    from allele_weaver.pbpk_dosing import simulate_pbpk_regimen

    sim = simulate_pbpk_regimen(drug, DosingRegimen(30.0, 8.0, 3), activity_score=2.0)
    curve = render_pk_curve(sim, drug)
    curve.encode("ascii")


def test_benchmark_suite_high_accuracy() -> None:
    res = run_benchmark_suite(seed=42, trials_per_case=2)
    assert res["total_trials"] == 20
    assert res["diplotype_accuracy"] >= 0.85
    assert res["cnv_accuracy"] >= 0.85


def test_cli_all_six_subcommands(tmp_path) -> None:
    call_json = tmp_path / "call.json"
    dose_csv = tmp_path / "dose.csv"
    bench_json = tmp_path / "bench.json"
    demo_json = tmp_path / "demo.json"

    assert cli_main(["phase", "--gene", "CYP2D6", "--diplotype", "*1/*4"]) == 0
    assert cli_main(["call", "--gene", "CYP2D6", "--diplotype", "*10/*36", "--json-out", str(call_json)]) == 0
    assert cli_main(["dose", "--drug", "warfarin", "--activity-score", "1.0", "--optimize", "--csv-out", str(dose_csv)]) == 0
    assert cli_main(["popgen", "--gene", "CYP2C19", "--n-samples", "80"]) == 0
    assert cli_main(["benchmark", "--trials", "1", "--json-out", str(bench_json)]) == 0
    assert cli_main(["demo", "--gene", "CYP2C9", "--diplotype", "*1/*3", "--drug", "warfarin", "--uq-draws", "15", "--json-out", str(demo_json)]) == 0

    assert call_json.exists()
    assert dose_csv.exists()
    assert bench_json.exists()
    demo_data = json.loads(demo_json.read_text(encoding="utf-8"))
    assert demo_data["gene_symbol"] == "CYP2C9"
