from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class GraphAdapter(ABC):
    @abstractmethod
    def export(self, vertices: pl.DataFrame, edges: pl.DataFrame, directed: bool = False) -> Any:
        pass
