class BackendProxy:
    def __init__(self, graph, backend_name):
        from .manager import ensure_materialized

        self._backend = ensure_materialized(backend_name, graph)

    def __getattr__(self, name):
        # Try backend-level function (e.g., networkx.shortest_path)
        fn = getattr(self._backend["module"], name, None)
        if fn:

            def wrapped(*args, **kwargs):
                return fn(self._backend["graph"], *args, **kwargs)

            return wrapped

        # Otherwise forward attribute to the backend graph itself
        return getattr(self._backend["graph"], name)
