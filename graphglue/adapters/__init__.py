from .networkx import *
from .igraph import *
from .graphtool import *
from .corneto import *

__all__ = ["networkx", "igraph"] # ..., "igraph", "graphtool", "corneto"]

# N.B. Oomit adapters.* from __all__ if you expect users to use G.to_networkx() rather than directly touching graphglue.adapters.network

"""
Standardize via adapter pattern with shared interface + fallback to edge list + attributes.
Support common exchange formats (e.g., via GraphML) for indirect conversion.
Use reflection/meta-inspection to auto-map foreign structures to internal schema.
"""

"""
Add caching for repeated conversions if structure unchanged.
Track deltas for selective sync back (esp. for subgraphs).
Enable partial proxies: wrap only relevant substructures instead of whole graph.
For read-only ops: avoid sync entirely, document immutability contract.
"""