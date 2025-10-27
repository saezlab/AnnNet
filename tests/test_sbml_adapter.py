import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from annnet.adapters.sbml_adapter import _graph_from_stoich, from_cobra_model
from annnet.core import Graph


class TestSBMLAdapter(unittest.TestCase):
    def test_graph_from_stoich_preserves_signs_and_sets(self):
        # Toy network:
        # R1: A + 2 B -> C
        # R2: C -> A
        mets = ["A", "B", "C"]
        rxns = ["R1", "R2"]
        S = np.array(
            [
                [-1, +1],  # A
                [-2, 0],  # B
                [+1, -1],  # C
            ],
            dtype=float,
        )

        G = Graph(directed=True)
        G = _graph_from_stoich(S, mets, rxns, graph=G, preserve_stoichiometry=True)

        # Vertices
        self.assertGreaterEqual(G.num_vertices, 3)

        # Edges present
        self.assertEqual(G.num_edges, 2)
        self.assertIn("R1", G.edge_to_idx)
        self.assertIn("R2", G.edge_to_idx)

        # Hyperedge definitions (head = products, tail = reactants)
        hdef_R1 = G.hyperedge_definitions["R1"]
        self.assertTrue(hdef_R1["directed"])
        self.assertEqual(hdef_R1["head"], {"C"})
        self.assertEqual(hdef_R1["tail"], {"A", "B"})

        # If we have per-vertex coefficients, incidence should reflect -2 for B
        col_R1 = G.edge_to_idx["R1"]
        row_A = G.entity_to_idx["A"]
        row_B = G.entity_to_idx["B"]
        row_C = G.entity_to_idx["C"]
        self.assertAlmostEqual(G._matrix[row_A, col_R1], -1.0)
        self.assertAlmostEqual(G._matrix[row_B, col_R1], -2.0)
        self.assertAlmostEqual(G._matrix[row_C, col_R1], +1.0)

    def test_cobra_integration_is_optional(self):
        try:
            import cobra  # noqa: F401
        except Exception:
            self.skipTest("COBRApy not installed in this environment")
        # Build a micro COBRA model in-memory if cobra is available
        from cobra import Metabolite, Model, Reaction

        model = Model("toy")
        A = Metabolite("A")
        B = Metabolite("B")
        C = Metabolite("C")

        R1 = Reaction("R1")
        R1.lower_bound = 0
        R1.upper_bound = 1000
        R1.add_metabolites({A: -1, B: -2, C: 1})

        R2 = Reaction("R2")
        R2.lower_bound = 0
        R2.upper_bound = 1000
        R2.add_metabolites({C: -1, A: 1})

        model.add_reactions([R1, R2])

        G = from_cobra_model(model, graph=Graph(directed=True))
        self.assertEqual(G.num_edges, 2)
        self.assertIn("R1", G.edge_to_idx)

    def test_boundary_reactions(self):
        import numpy as np

        from annnet.adapters.sbml_adapter import (
            BOUNDARY_SINK,
            BOUNDARY_SOURCE,
            _graph_from_stoich,
        )

        mets = ["A"]
        rxns = ["deg", "syn"]
        S = np.array([[-1.0, +1.0]])  # A degrades (col0) and is synthesized (col1)
        G = _graph_from_stoich(
            S, mets, rxns, graph=Graph(directed=True), preserve_stoichiometry=True
        )
        h_deg = G.hyperedge_definitions["deg"]
        h_syn = G.hyperedge_definitions["syn"]
        assert h_deg["tail"] == {"A"} and h_deg["head"] == {BOUNDARY_SINK}
        assert h_syn["tail"] == {BOUNDARY_SOURCE} and h_syn["head"] == {"A"}


if __name__ == "__main__":
    unittest.main()
