import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import pytest
from graphglue.adapters.json_adapter import to_json, from_json  # JSON (JavaScript Object Notation)
from graphglue.adapters.GraphDir_Parquet_adapter import write_parquet_graphdir, read_parquet_graphdir  # Parquet (columnar storage)
from graphglue.adapters.dataframe_adapter import to_dataframes, from_dataframes  # DF (DataFrame)
import time

class TestPerformance:
    """Optional performance comparison tests."""

    @pytest.mark.slow
    def test_adapter_speed_comparison(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        results = {}
        start = time.time()
        to_json(G, tmpdir_fixture / "perf.json")
        from_json(tmpdir_fixture / "perf.json")
        results['JSON'] = time.time() - start

        start = time.time()
        write_parquet_graphdir(G, tmpdir_fixture / "perf_dir")
        H = read_parquet_graphdir(tmpdir_fixture / "perf_dir")
        # optional sanity:
        assert H.number_of_edges() == G.number_of_edges()
        results['Parquet'] = time.time() - start

        start = time.time()
        dfs = to_dataframes(G)
        from_dataframes(**dfs)
        results['DataFrame'] = time.time() - start

        print("\nAdapter Performance (seconds):")
        for adapter, elapsed in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {adapter}: {elapsed:.4f}s")
