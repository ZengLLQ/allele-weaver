"""Module execution entry point for `python -m allele_weaver`."""

from __future__ import annotations

import sys

from allele_weaver.cli import main

if __name__ == "__main__":
    sys.exit(main())
