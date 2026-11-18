"""Example 04: Multi-gene cohort benchmark for MEC phasing, CNV recall, and star-allele calling."""

from __future__ import annotations

from allele_weaver.pipeline import run_benchmark_suite


def main() -> None:
    report = run_benchmark_suite(seed=42, trials_per_case=3)
    print("=== ALLELEWEAVER BENCHMARK SUMMARY ===")
    print(
        f"Trials={report['total_trials']} | "
        f"Diplotype Accuracy={report['diplotype_accuracy'] * 100:.1f}% | "
        f"CNV Accuracy={report['cnv_accuracy'] * 100:.1f}% | "
        f"Mean Switch Error={report['mean_switch_error_rate']:.4f}"
    )
    for row in report["cases"]:
        print(
            f"  {row['gene']:<8s} | {row['true_diplotype']:<10s} | "
            f"CNV={row['expected_cnv']:<11s} | Dip={row['diplotype_recall']} | "
            f"CNV_Rec={row['cnv_recall']}"
        )


if __name__ == "__main__":
    main()
