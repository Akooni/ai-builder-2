"""Unit tests for CPU ↔ motherboard socket compatibility (SearchEngine._compatible_mb)."""

from __future__ import annotations

import unittest

import pandas as pd

from search_engine import SearchEngine


def _minimal_tables(cpus: list[dict], mbs: list[dict]) -> dict[str, pd.DataFrame]:
    """Build a minimal catalog so SearchEngine can be constructed; only CPU/MB rows are used."""
    empty = pd.DataFrame()
    return {
        "CPUs": pd.DataFrame(cpus),
        "MBs": pd.DataFrame(mbs),
        "RAMs": empty,
        "Storage": empty,
        "GPUs": empty,
        "PSUs": empty,
    }


class TestCpuMotherboardSocketCompatibility(unittest.TestCase):
    """Sockets must match after trimming whitespace (see search_engine.SearchEngine._compatible_mb)."""

    def test_matching_sockets_am4(self) -> None:
        tables = _minimal_tables(
            [{"socket": "AM4", "name": "Test CPU"}],
            [{"socket": "AM4", "name": "Test MB"}],
        )
        b = SearchEngine(tables)
        self.assertTrue(b._compatible_mb(0, 0))

    def test_matching_sockets_lga1700(self) -> None:
        tables = _minimal_tables(
            [{"socket": "LGA1700", "name": "Intel CPU"}],
            [{"socket": "LGA1700", "name": "Intel MB"}],
        )
        b = SearchEngine(tables)
        self.assertTrue(b._compatible_mb(0, 0))

    def test_mismatched_sockets_am4_vs_am5(self) -> None:
        tables = _minimal_tables(
            [{"socket": "AM4", "name": "AM4 CPU"}],
            [{"socket": "AM5", "name": "AM5 MB"}],
        )
        b = SearchEngine(tables)
        self.assertFalse(b._compatible_mb(0, 0))

    def test_mismatched_sockets_am4_vs_lga1700(self) -> None:
        tables = _minimal_tables(
            [{"socket": "AM4", "name": "AMD CPU"}],
            [{"socket": "LGA1700", "name": "Intel MB"}],
        )
        b = SearchEngine(tables)
        self.assertFalse(b._compatible_mb(0, 0))

    def test_whitespace_around_socket_is_ignored(self) -> None:
        tables = _minimal_tables(
            [{"socket": "  AM4  ", "name": "CPU"}],
            [{"socket": "AM4", "name": "MB"}],
        )
        b = SearchEngine(tables)
        self.assertTrue(b._compatible_mb(0, 0))

        tables2 = _minimal_tables(
            [{"socket": "LGA1700", "name": "CPU"}],
            [{"socket": "\tLGA1700\n", "name": "MB"}],
        )
        b2 = SearchEngine(tables2)
        self.assertTrue(b2._compatible_mb(0, 0))

    def test_socket_comparison_is_case_sensitive(self) -> None:
        """Implementation uses str equality after strip only (no casefold)."""
        tables = _minimal_tables(
            [{"socket": "am4", "name": "lower"}],
            [{"socket": "AM4", "name": "upper"}],
        )
        b = SearchEngine(tables)
        self.assertFalse(b._compatible_mb(0, 0))

    def test_correct_pair_among_multiple_rows(self) -> None:
        tables = _minimal_tables(
            [
                {"socket": "AM4", "name": "CPU0"},
                {"socket": "AM5", "name": "CPU1"},
            ],
            [
                {"socket": "AM5", "name": "MB0"},
                {"socket": "AM4", "name": "MB1"},
            ],
        )
        b = SearchEngine(tables)
        self.assertTrue(b._compatible_mb(0, 1))  # AM4 + AM4
        self.assertTrue(b._compatible_mb(1, 0))  # AM5 + AM5
        self.assertFalse(b._compatible_mb(0, 0))  # AM4 CPU + AM5 MB
        self.assertFalse(b._compatible_mb(1, 1))  # AM5 CPU + AM4 MB


if __name__ == "__main__":
    unittest.main()
