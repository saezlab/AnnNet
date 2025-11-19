# tests/test_plotting.py
import re
import unittest
import warnings

import numpy as np

# Silence harmless NumPy warning on some builds
warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from annnet.core.graph import Graph
from annnet.utils import plotting


class TestHelpers(unittest.TestCase):
    def test_normalize_basic_and_bounds(self):
        x = plotting._normalize([0, 5, 10])
        self.assertTrue(np.allclose(x, [0.0, 0.5, 1.0], atol=1e-12))
        x2 = plotting._normalize([5, 7.5, 10], lo=0, hi=10)
        self.assertTrue(np.allclose(x2, [0.5, 0.75, 1.0], atol=1e-12))
        self.assertEqual(plotting._normalize([]).size, 0)

    def test_greyscale_format_and_clamp(self):
        self.assertRegex(plotting._greyscale(0.0), r"^#[0-9a-f]{6}$")
        self.assertEqual(plotting._greyscale(0.0), "#000000")
        self.assertEqual(plotting._greyscale(1.0), "#ffffff")
        self.assertEqual(plotting._greyscale(-5.0), "#000000")
        self.assertEqual(plotting._greyscale(5.0), "#ffffff")


class TestPlottingWithRealGraph(unittest.TestCase):
    def setUp(self):
        g = Graph()
        # vertices with labels
        g.add_vertex("A", label="alpha")
        g.add_vertex("B", label="beta")
        g.add_vertex("C", label="gamma")
        # edges with attributes embedded (non-reserved keys persist)
        e1 = g.add_edge("A", "B", weight=2.0, interaction=+1, type="activation")
        e2 = g.add_edge("B", "C", weight=-1.0, interaction=-1)
        e3 = g.add_hyperedge(head=["A", "B"], tail=["C"], weight=0.5, interaction=+1)

        # Per-slice override for e1
        g.add_slice("Lw")
        g.set_edge_slice_attrs("Lw", e1, weight=5.0)

        self.g = g
        # Sanity: plotting relies on these
        self.assertGreaterEqual(self.g.number_of_edges(), 2)
        self.assertTrue(hasattr(self.g, "idx_to_edge"))
        self.assertTrue(hasattr(self.g, "get_edge"))

    def test_build_vertex_labels(self):
        labels = plotting.build_vertex_labels(self.g, key="label")
        self.assertEqual(labels["A"], "alpha")
        self.assertEqual(labels["B"], "beta")
        # fallback to id if key missing (none missing here)
        labels_ids = plotting.build_vertex_labels(self.g, key=None)
        self.assertEqual(labels_ids["A"], "A")

    def test_build_edge_labels(self):
        lbls = plotting.build_edge_labels(self.g, use_weight=True, extra_keys=["type"], slice=None)
        # keys are edge indices (0..m-1)
        self.assertIn(0, lbls)
        self.assertRegex(lbls[0], r"w=2(\.0+)?")
        # slice-aware override
        lbls_Lw = plotting.build_edge_labels(self.g, use_weight=True, extra_keys=[], slice="Lw")
        self.assertRegex(lbls_Lw[0], r"w=5(\.0+)?")

    def test_edge_style_signed(self):
        styles = plotting.edge_style_from_weights(self.g, color_mode="signed")
        self.assertIn(0, styles)
        self.assertIn("color", styles[0])
        # Signed mapping: positive -> firebrick4, negative -> dodgerblue4
        colors = [styles[i]["color"] for i in sorted(styles)]
        self.assertIn("firebrick4", colors)
        self.assertIn("dodgerblue4", colors)


class TestBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # optional deps
        try:
            import graphviz  # noqa: F401

            cls.HAS_GRAPHVIZ = True
        except Exception:
            cls.HAS_GRAPHVIZ = False
        try:
            import pydot  # noqa: F401

            cls.HAS_PYDOT = True
        except Exception:
            cls.HAS_PYDOT = False

    def setUp(self):
        g = Graph()
        g.add_vertex("A", label="alpha")
        g.add_vertex("B", label="beta")
        g.add_vertex("C", label="gamma")
        g.add_edge("A", "B", weight=2.0, interaction=+1)
        g.add_edge("B", "C", weight=-1.0, interaction=-1)
        g.add_hyperedge(head=["A", "B"], tail=["C"], weight=0.5, interaction=+1)
        self.g = g

    def test_to_graphviz_builds_dot_source_when_available(self):
        if not self.HAS_GRAPHVIZ:
            self.skipTest("graphviz package not installed")
        Gv = plotting.to_graphviz(self.g, layout="dot", orphan_edges=True)
        src = Gv.source
        self.assertIn("A", src)
        self.assertIn("B", src)
        self.assertIn("C", src)
        self.assertRegex(src, r"A\s*->\s*B")
        self.assertRegex(src, r"B\s*->\s*C")

    def test_to_pydot_builds_graph_when_available(self):
        if not self.HAS_PYDOT:
            self.skipTest("pydot package not installed")
        Gd = plotting.to_pydot(self.g, orphan_edges=True)
        self.assertGreaterEqual(len(Gd.get_edges()), 2)

    def test_plot_graphviz_and_labels_when_available(self):
        if not self.HAS_GRAPHVIZ:
            self.skipTest("graphviz package not installed")
        Gv = plotting.plot(
            self.g, backend="graphviz", show_vertex_labels=True, show_edge_labels=True
        )
        src = Gv.source
        self.assertRegex(src, r"\bA\s*\[label=A\b")
        self.assertRegex(src, r"\bB\s*\[label=B\b")
        self.assertRegex(src, r"\bC\s*\[label=C\b")
        self.assertRegex(src, r"->")  # edges present

    def test_plot_pydot_and_labels_when_available(self):
        if not self.HAS_PYDOT:
            self.skipTest("pydot package not installed")
        Gd = plotting.plot(self.g, backend="pydot", show_vertex_labels=True, show_edge_labels=True)
        labels = [e.get("label") for e in Gd.get_edges()]
        self.assertTrue(any(lbl and re.search(r"w=", lbl) for lbl in labels))

    def test_render_type_error_for_unknown_object(self):
        with self.assertRaises(TypeError):
            plotting.render(object(), "out.svg", format="svg")


if __name__ == "__main__":
    unittest.main()
