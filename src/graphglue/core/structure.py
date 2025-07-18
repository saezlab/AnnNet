from enum import Enum

class EdgeType(str, Enum):
    """Edge type (DIRECTED, UNDIRECTED).

    Attributes:
        DIRECTED: Represents a directed edge
        UNDIRECTED: Represents an undirected edge
    """

    DIRECTED = "directed"
    UNDIRECTED = "undirected"

"""
Use a uniform internal representation based on edge-centric views with pluggable structure handlers.
Hybrid storage: sparse matrices + typed metadata layers for high performance.
"""