import hashlib


MAX_DEPTH = 6


class FakeRLock:
    """A fake class to replace threading.RLock during serialization"""

    def acquire(self):
        pass

    def release(self):
        pass


def recursive_hash(value, depth=0, ignore_params=[]):
    """Hash primitives recursively with maximum depth."""
    if depth > MAX_DEPTH:
        return hashlib.sha256("max_depth_reached".encode()).hexdigest()

    if isinstance(value, (int, float, str, bool, bytes)):
        return hashlib.sha256(str(value).encode()).hexdigest()
    elif isinstance(value, (list, tuple)):
        return hashlib.sha256(
            "".join([recursive_hash(item, depth + 1, ignore_params) for item in value]).encode()
        ).hexdigest()
    elif isinstance(value, dict):
        return hashlib.sha256(
            "".join(
                [
                    recursive_hash(key, depth + 1, ignore_params)
                    + recursive_hash(val, depth + 1, ignore_params)
                    for key, val in value.items()
                    if key not in ignore_params
                ]
            ).encode()
        ).hexdigest()
    elif hasattr(value, "__dict__") and value.__class__.__name__ not in ignore_params:
        return recursive_hash(value.__dict__, depth, ignore_params)
    else:
        return hashlib.sha256("unknown".encode()).hexdigest()


def shorten_filename(filename, length, hash_length=16):
    """
    Shorten the filename to a fixed length, keeping it unique by collapsing partly into a hash.
    Keeps the start and end of the filename for readability.
    """
    assert length > hash_length + 2, "Length should be greater than hash length + 2"
    if len(filename) > length:
        hash_part = hashlib.sha256(filename.encode()).hexdigest()[:hash_length]
        filename = "{}_{}_{}".format(
            filename[: length // 2 - hash_length // 2 - 1],
            hash_part,
            filename[-length // 2 + hash_length // 2 + 1 :],
        )
    return filename
