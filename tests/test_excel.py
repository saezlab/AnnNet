import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Hard dependency for this adapter:
import importlib.util
import json
import os
import re
import tempfile
import unittest

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if _PANDAS_AVAILABLE:
    import pandas as pd


# Utilities for robust assertions
import polars as pl

from graphglue.io.excel import load_excel_to_graph


def _colmap(df: pl.DataFrame):
    return {c.lower(): c for c in df.columns}


_SEP = re.compile(r"[|,; ]+")


def _explode_cell(x):
    """Return list of tokens from members/head/tail cell (handles Series/list/JSON/string)."""
    if x is None:
        return []
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(x, (list, tuple, set)):
        return [str(y) for y in x]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(y) for y in (arr if isinstance(arr, (list, tuple)) else [arr])]
        except Exception:
            pass
    return [p for p in _SEP.split(s) if p]


@unittest.skipUnless(_PANDAS_AVAILABLE, "pandas is required to read/write Excel for this adapter")
class TestExcelIO(unittest.TestCase):
    def _write_excel_temp(self, df_map, suffix=".xlsx"):
        """df_map: dict[str, pandas.DataFrame]  (sheet_name -> DF)
        Returns path to a temporary Excel file. Caller removes it.
        """
        # Try engines in order; if none available, skip
        engines = [None, "openpyxl", "xlsxwriter"]
        last_err = None
        for eng in engines:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.close()  # we'll write using pandas
                with (
                    pd.ExcelWriter(tmp.name, engine=eng)
                    if eng
                    else pd.ExcelWriter(tmp.name) as writer
                ):
                    for sheet, df in df_map.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)
                return tmp.name
            except Exception as e:
                last_err = e
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        self.skipTest(f"No Excel writer engine available for pandas: {last_err!r}")

    def test_edge_list_excel_roundtrip(self):
        # Prepare edge-list sheet
        df = pd.DataFrame(
            {
                "source": ["A", "B"],
                "target": ["B", "C"],
                "weight": [1.0, 2.0],
                "directed": [
                    True,
                    False,
                ],  # importer may coerce; we won't assert exact directedness
                "layer": ["L1", "L1"],
            }
        )
        path = self._write_excel_temp({"Edges": df})

        try:
            G = load_excel_to_graph(path, schema="edge_list")
            ev = G.edges_view()
            self.assertEqual(ev.shape[0], 2)
            cols = _colmap(ev)
            src = cols.get("source") or cols.get("src") or cols.get("u") or cols.get("from")
            tgt = cols.get("target") or cols.get("dst") or cols.get("v") or cols.get("to")
            wcol = (
                cols.get("effective_weight")
                or cols.get("global_weight")
                or cols.get("weight")
                or cols.get("w")
                or cols.get("edge_weight")
            )
            self.assertIsNotNone(src)
            self.assertIsNotNone(tgt)
            self.assertIsNotNone(wcol)
            rows = [tuple(r) for r in ev.select([src, tgt, wcol]).iter_rows()]
            self.assertIn(("A", "B", 1.0), rows)
            self.assertIn(("B", "C", 2.0), rows)
        finally:
            os.unlink(path)

    def test_hyperedge_excel_roundtrip(self):
        # Prepare hyperedge sheet (members as pipe-separated string)
        df = pd.DataFrame(
            {
                "members": ["A|B|C"],
                "weight": [3.0],
                "layer": ["L2"],
            }
        )
        path = self._write_excel_temp({"Hyper": df})

        try:
            G = load_excel_to_graph(path, schema="hyperedge")
            ev = G.edges_view()
            self.assertGreaterEqual(ev.shape[0], 1)
            cols = _colmap(ev)
            kind = cols.get("kind")
            self.assertIsNotNone(kind)
            kind_val = (ev.select(kind).to_series()[0] or "").lower()
            self.assertIn(kind_val, ("hyper", "hyperedge"))

            # members OR head/tail must exist
            members = cols.get("members")
            head = cols.get("head")
            tail = cols.get("tail")
            if members:
                raw = ev.select(members).to_series()[0]
                parts = set(_explode_cell(raw))
            elif head and tail:
                h_raw = ev.select(head).to_series()[0]
                t_raw = ev.select(tail).to_series()[0]
                parts = set(_explode_cell(h_raw)) | set(_explode_cell(t_raw))
            else:
                self.fail(
                    "No hyperedge columns ('members' or 'head'/'tail') found in edges_view()."
                )

            self.assertEqual(parts, {"A", "B", "C"})
        finally:
            os.unlink(path)

    def test_auto_schema_detection(self):
        # Only source/target provided → should infer edge_list
        df = pd.DataFrame({"source": ["U"], "target": ["V"]})
        path = self._write_excel_temp({"Sheet1": df})
        try:
            G = load_excel_to_graph(path, schema="auto")
            ev = G.edges_view()
            self.assertEqual(ev.shape[0], 1)
            cols = _colmap(ev)
            self.assertIn("source", {c.lower() for c in ev.columns})
            self.assertIn("target", {c.lower() for c in ev.columns})
        finally:
            os.unlink(path)

    def test_sheet_selection(self):
        # Two sheets; we will load the non-first one by name
        df1 = pd.DataFrame({"source": ["X"], "target": ["Y"]})
        df2 = pd.DataFrame({"source": ["P"], "target": ["Q"]})
        path = self._write_excel_temp({"First": df1, "Second": df2})
        try:
            G = load_excel_to_graph(path, schema="edge_list", sheet="Second")
            ev = G.edges_view()
            self.assertEqual(ev.shape[0], 1)
            cols = _colmap(ev)
            src = cols.get("source") or "source"
            tgt = cols.get("target") or "target"

            # FIX: iter_rows() is a generator → grab the first item safely
            row_iter = ev.select([src, tgt]).iter_rows()
            first_row = next(row_iter, None)
            self.assertIsNotNone(first_row, "No rows loaded from sheet 'Second'")
            tup = tuple(first_row)

            self.assertEqual(tup, ("P", "Q"))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
