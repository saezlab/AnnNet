from __future__ import annotations
import tempfile
import pathlib
from typing import Optional, Union

import polars as pl

from ..core.incgraph import IncidenceGraph


def load_excel_to_graph(
    path: Union[str, pathlib.Path],
    graph=None,
    schema: str = "auto",
    sheet: Optional[str] = None,
    default_layer=None,
    default_directed=None,
    default_weight: float = 1.0,
    **kwargs,
):
    """
    Load an Excel (.xlsx/.xls) file by converting it internally to CSV, then building a graph.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.
    graph : IncidenceGraph, optional
        Existing graph instance. If None, a new one is created.
    schema : {'auto', 'edge_list', 'hyperedge', 'incidence', 'adjacency', 'lil'}, default 'auto'
        Graph schema to assume or infer.
    sheet : str, optional
        Sheet name to load. Defaults to the first sheet.
    default_layer : str, optional
        Default layer name if not present.
    default_directed : bool, optional
        Default directedness if not present or inferrable.
    default_weight : float, default 1.0
        Default weight if no weight column exists.
    **kwargs
        Extra keyword arguments passed to the graph constructor.

    Returns
    -------
    IncidenceGraph
        The created or augmented graph.

    Notes
    -----
    - This function **does not require `fastexcel` or `openpyxl`**.
    - The Excel is read once into memory and written to a temporary CSV, then processed with the CSV loader.
    - Supported formats and schemas are identical to `load_csv_to_graph`.
    """
    path = pathlib.Path(path)

    # Convert Excel → temporary CSV
    # Using pandas for one-time conversion (NOT a package dependency if user has pandas in the notebook)
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Excel support requires `pandas` at runtime for conversion. "
            "Install it or convert the file to CSV manually."
        ) from e

    # Read sheet (default first)
    data = pd.read_excel(path, sheet_name=sheet)

    # If multiple sheets, pick the first one (or allow user to choose)
    if isinstance(data, dict):
        if sheet is None:
            # Pick the first sheet if user didn't specify
            sheet_name, df = next(iter(data.items()))
        else:
            df = data[sheet]
    else:
        df = data

    # Now convert to CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
        df.to_csv(tmp_path, index=False)


    # Pass the temporary CSV into the existing loader
    from .csv import load_csv_to_graph
    G = load_csv_to_graph(
        tmp_path,
        graph=graph,
        schema=schema,
        default_layer=default_layer,
        default_directed=default_directed,
        default_weight=default_weight,
        **kwargs,
    )

    return G