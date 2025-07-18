Getting Started
===============

To use GraphGlue, install the package and import the core API:

.. code-block:: bash

   poetry install --extras "networkx igraph"
   poetry run gg-demo

Example:

.. code-block:: python

   from graphglue import Graph
   g = Graph()
   g.add_node("X")
   print(g.nodes)

See the :doc:`modules` for more details.

