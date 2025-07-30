class _State:
    def __init__(self):
        self.version = 0
        self._backend_cache = {}

    def dirty_since(self, version: int) -> bool:
        return self.version > version
