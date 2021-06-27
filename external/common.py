"""
Common code that is present in pesq and stoi
"""

from functools import wraps
from multiprocessing.pool import Pool
from typing import Callable

from numpy import ndarray


def evaluate(func: Callable) -> Callable:
    "Evaluate using a function"

    @wraps(evaluate)
    def evaluator(ref: ndarray, deg: ndarray, sr: int, pool: Pool = None) -> float:
        length = len(ref)
        assert length == len(deg)

        return (
            sum(
                pool.starmap(
                    func=func, iterable=((r, d, sr) for (r, d) in zip(ref, deg))
                )
            )
            / length
        )

    return evaluator
