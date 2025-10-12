from collections import OrderedDict
from typing import Any, Tuple

class LRU:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key: Tuple[Any, ...]):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def set(self, key: Tuple[Any, ...], value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)
