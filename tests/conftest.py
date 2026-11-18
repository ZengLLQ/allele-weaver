"""Shared pytest fixtures for AlleleWeaver unit, invariant, and integration tests."""

from __future__ import annotations

import pytest

from allele_weaver.catalog import SyntheticCohortGenerator, get_pharmacogene_locus
from allele_weaver.models import PharmacogeneLocus


@pytest.fixture
def cyp2d6_locus() -> PharmacogeneLocus:
    """Return fresh CYP2D6 locus definition."""
    return get_pharmacogene_locus("CYP2D6")


@pytest.fixture
def cyp2c9_locus() -> PharmacogeneLocus:
    """Return fresh CYP2C9 locus definition."""
    return get_pharmacogene_locus("CYP2C9")


@pytest.fixture
def cyp2c19_locus() -> PharmacogeneLocus:
    """Return fresh CYP2C19 locus definition."""
    return get_pharmacogene_locus("CYP2C19")


@pytest.fixture
def generator() -> SyntheticCohortGenerator:
    """Return deterministic synthetic cohort generator."""
    return SyntheticCohortGenerator(seed=2026)
