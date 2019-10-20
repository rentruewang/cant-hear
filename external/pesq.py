from multiprocessing.pool import ThreadPool

import numpy as np

from pypesq import pesq as _pesq


def pesq(ref: np.ndarray, deg: np.ndarray, sr: int, pool: ThreadPool = None) -> float:
    length = len(ref)
    assert length == len(deg)
    if pool is not None:
        return (
            sum(
                pool.starmap(
                    func=_pesq, iterable=((r, d, sr) for (r, d) in zip(ref, deg))
                )
            )
            / length
        )
    else:
        return sum(_pesq(r, d, sr) for (r, d) in zip(ref, deg)) / length
