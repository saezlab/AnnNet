import hashlib
import json
from collections.abc import Callable, Iterable
from itertools import filterfalse
from typing import Any, TypeVar

T = TypeVar("T")


def canonicalize(obj):
    """Recursively convert an object into a JSON-serializable structure
    that is independent of internal ordering.
    """
    if isinstance(obj, dict):
        # Convert dictionary keys to strings and sort the keys
        return {
            str(key): canonicalize(obj[key]) for key in sorted(obj.keys(), key=lambda x: str(x))
        }
    elif isinstance(obj, (list, tuple)):
        # Recursively canonicalize each element in the list or tuple
        return [canonicalize(item) for item in obj]
    elif isinstance(obj, set):
        # Convert sets to a sorted list (sorting based on JSON string representation)
        return sorted(
            [canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True),
        )
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # For non-standard objects, try using the __dict__ attribute if available
        if hasattr(obj, "__dict__"):
            return canonicalize(obj.__dict__)
        else:
            # Fall back to a string representation
            return str(obj)


def obj_canonicalized_hash(obj) -> str:
    # First canonicalize the object
    canonical_obj = canonicalize(obj)
    # Serialize the canonical object to a JSON string.
    # 'sort_keys=True' ensures consistent key order,
    # and separators remove unnecessary whitespace.
    obj_serialized = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    # Compute the SHA256 hash of the serialized bytes
    hash_obj = hashlib.sha256()
    hash_obj.update(obj_serialized)
    return hash_obj.hexdigest()


def unique_iter(iterable: Iterable[T], key: Callable[[T], Any] | None = None) -> Iterable[T]:
    # Based on https://iteration-utilities.readthedocs.io/en/latest/generated/unique_everseen.html
    seen: set[Any] = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
