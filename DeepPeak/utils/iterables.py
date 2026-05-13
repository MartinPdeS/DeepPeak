from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """
    Yield successive batches of size `n` from `iterable`.

    This mirrors `itertools.batched`, which is only available in Python 3.12+.
    """
    if n < 1:
        raise ValueError("n must be at least one")

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch
