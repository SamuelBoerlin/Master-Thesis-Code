import hashlib
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer


def derive_seed(*seed_data: "ReadableBuffer") -> int:
    assert len(seed_data) > 0

    digest = hashlib.sha1(seed_data[0])

    if len(seed_data) > 1:
        for seed_datum in seed_data[1:]:
            digest.update(seed_datum)

    seed = int(digest.hexdigest(), 16)

    if seed < 0:
        seed = -seed

    seed %= 2**32 - 1

    return seed


def derive_rng(*seed_data: "ReadableBuffer") -> Generator:
    return np.random.default_rng(seed=derive_seed(*seed_data))


def random_seed(generator: Generator) -> int:
    return generator.integers(low=0, high=2**32 - 1)


def random_seed_bytes(generator: Generator) -> bytes:
    return str(random_seed(generator)).encode()


T = TypeVar("T", bound=Any)


def discrete_distribution(
    histogram_x: NDArray[T],
    histogram_y: NDArray,
) -> Callable[[Generator], T]:
    if len(histogram_y.shape) != 1:
        raise ValueError("len(y_histogram.shape) != 1")

    if histogram_x.shape[0] != histogram_y.shape[0]:
        raise ValueError("x_histogram.shape[0] != y_histogram.shape[0]")

    cdf = np.cumsum(histogram_y / np.sum(histogram_y))

    def distribution(generator: Generator) -> T:
        x = generator.random()
        return histogram_x[np.argmax(cdf > x)]

    return distribution


def sample_discrete_distribution(
    generator: Generator,
    histogram_x: NDArray[T],
    histogram_y: NDArray,
) -> T:
    return discrete_distribution(histogram_x, histogram_y)(generator)
