"""Bioinformatics I/O parsers and structured telemetry exporters.

Supports self-contained parsing and serialization without external dependencies:
- VCF v4.2 variant records with phased ('|') and unphased ('/') GT, AD, DP, GQ
- SAM-style CIGAR alignment strings for read-span and indel parsing
- Tab-delimited read-fragment matrices with Phred ASCII-33 quality strings
- JSON clinical reports and CSV pharmacokinetic time-course telemetry
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from allele_weaver.models import (
    AlleleWeaverError,
    PKSimulationResult,
    ReadFragment,
    VariantSite,
)

_CIGAR_TOKEN_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_cigar(cigar: str) -> List[Tuple[int, str]]:
    """Parse a SAM CIGAR string (e.g. '75M2I40M') into (length, op) tuples.

    Raises:
        AlleleWeaverError: If the CIGAR string is malformed or empty.
    """
    if not cigar or cigar == "*":
        raise AlleleWeaverError(f"Empty or unmapped CIGAR string: '{cigar}'")
    tokens = _CIGAR_TOKEN_RE.findall(cigar)
    reconstructed = "".join(f"{length}{op}" for length, op in tokens)
    if reconstructed != cigar:
        raise AlleleWeaverError(f"Malformed CIGAR string: '{cigar}'")
    return [(int(length), op) for length, op in tokens]


def cigar_reference_span(cigar: str) -> int:
    """Compute reference genomic span (bp) consumed by a CIGAR alignment."""
    ref_ops = {"M", "D", "N", "=", "X"}
    return sum(length for length, op in parse_cigar(cigar) if op in ref_ops)


def phred_char_to_q(ch: str) -> int:
    """Convert single Phred+33 ASCII character into integer base quality."""
    if len(ch) != 1:
        raise AlleleWeaverError(f"Expected single Phred character, got '{ch}'")
    q = ord(ch) - 33
    if q < 1 or q > 93:
        raise AlleleWeaverError(f"Phred quality out of ASCII-33 range [1, 93]: {q}")
    return q


def parse_vcf(text: str) -> List[VariantSite]:
    """Parse VCF v4.2 text into VariantSite records for the first sample column.

    Extracts GT, AD (comma-separated ref,alt depths), and DP when present in FORMAT.
    """
    sites: List[VariantSite] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            raise AlleleWeaverError(f"VCF line {line_no} has fewer than 8 columns")
        chrom, pos_s, vid, ref, alt, _qual, _flt, info = cols[:8]
        pos = int(pos_s)
        rsid = vid if vid != "." else f"{chrom}:{pos}"
        alt_primary = alt.split(",")[0]

        impact = "normal"
        paralog_rate = 0.0
        for item in info.split(";"):
            if item.startswith("IMPACT="):
                impact = item.split("=", 1)[1]
            elif item.startswith("PARALOG="):
                paralog_rate = float(item.split("=", 1)[1])

        gt = "0/0"
        ref_count = 20
        alt_count = 0
        if len(cols) >= 10:
            fmt_keys = cols[8].split(":")
            sample_vals = cols[9].split(":")
            fmt_map = dict(zip(fmt_keys, sample_vals))
            if "GT" in fmt_map and fmt_map["GT"] != "./.":
                gt = fmt_map["GT"]
            if "AD" in fmt_map and "," in fmt_map["AD"]:
                parts = fmt_map["AD"].split(",")
                ref_count = int(parts[0])
                alt_count = int(parts[1])
            elif "DP" in fmt_map and fmt_map["DP"].isdigit():
                dp = int(fmt_map["DP"])
                if "1" in gt and "0" in gt:
                    ref_count = dp // 2
                    alt_count = dp - ref_count
                elif gt in ("1/1", "1|1"):
                    ref_count = 0
                    alt_count = dp
                else:
                    ref_count = dp
                    alt_count = 0

        sites.append(
            VariantSite(
                rsid=rsid,
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt_primary,
                functional_impact=impact,
                ref_count=ref_count,
                alt_count=alt_count,
                genotype=gt,
                paralog_mismap_rate=paralog_rate,
            )
        )
    return sites


def write_vcf(sites: Iterable[VariantSite], sample_id: str = "PATIENT_001") -> str:
    """Serialize VariantSite records into valid VCF v4.2 text."""
    lines = [
        "##fileformat=VCFv4.2",
        "##source=AlleleWeaver",
        '##INFO=<ID=IMPACT,Number=1,Type=String,Description="Functional consequence">',
        '##INFO=<ID=PARALOG,Number=1,Type=Float,Description="Paralog mismapping fraction">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Total read depth">',
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}",
    ]
    for s in sites:
        info = f"IMPACT={s.functional_impact};PARALOG={s.paralog_mismap_rate:.4f}"
        sample = f"{s.genotype}:{s.ref_count},{s.alt_count}:{s.total_depth}"
        lines.append(
            f"{s.chrom}\t{s.pos}\t{s.rsid}\t{s.ref}\t{s.alt}\t99\tPASS\t{info}\tGT:AD:DP\t{sample}"
        )
    return "\n".join(lines) + "\n"


def parse_fragment_tsv(text: str) -> List[ReadFragment]:
    """Parse tab-delimited read fragments: read_id<TAB>mapq<TAB>idx:allele:phred_q,..."""
    fragments: List[ReadFragment] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise AlleleWeaverError(f"Fragment line {line_no} requires 3 tab-separated columns")
        read_id = parts[0]
        mapq = int(parts[1])
        obs: Dict[int, Tuple[int, int]] = {}
        for token in parts[2].split(","):
            if not token:
                continue
            fields = token.split(":")
            if len(fields) != 3:
                raise AlleleWeaverError(f"Malformed observation token '{token}' on line {line_no}")
            idx = int(fields[0])
            allele = int(fields[1])
            qual = int(fields[2])
            obs[idx] = (allele, qual)
        fragments.append(ReadFragment(read_id=read_id, observations=obs, mapping_quality=mapq))
    return fragments


def serialize_fragments_tsv(fragments: Iterable[ReadFragment]) -> str:
    """Serialize ReadFragment objects to tab-delimited fragment format."""
    lines = ["#read_id\tmapq\tobs(idx:allele:qual)"]
    for frag in fragments:
        obs_str = ",".join(
            f"{idx}:{allele}:{qual}" for idx, (allele, qual) in sorted(frag.observations.items())
        )
        lines.append(f"{frag.read_id}\t{frag.mapping_quality}\t{obs_str}")
    return "\n".join(lines) + "\n"


def export_pk_telemetry_csv(sim: PKSimulationResult, path: Path) -> None:
    """Export time-course PBPK/PD trajectory to CSV with physical units in header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "time_h",
            "c_plasma_mg_per_l",
            "c_hepatic_mg_per_l",
            "c_periph_mg_per_l",
            "c_metabolite_mg_per_l",
            "pd_effect",
        ])
        for row in zip(
            sim.time_h,
            sim.c_plasma_mg_per_l,
            sim.c_hepatic_mg_per_l,
            sim.c_periph_mg_per_l,
            sim.c_metabolite_mg_per_l,
            sim.pd_effect,
        ):
            writer.writerow([f"{v:.6f}" for v in row])


def export_clinical_json(payload: Dict[str, Any], path: Optional[Path] = None) -> str:
    """Serialize clinical report dictionary to deterministic formatted JSON."""
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    return serialized
