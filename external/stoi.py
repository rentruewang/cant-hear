from multiprocessing.pool import ThreadPool

import numpy as np

from pystoi.stoi import stoi as _stoi


def stoi(ref: np.ndarray, deg: np.ndarray, sr: int, pool: ThreadPool = None) -> float:
    length = len(ref)
    assert length == len(deg)
    if pool is not None:
        return (
            sum(
                pool.starmap(
                    func=_stoi, iterable=((r, d, sr) for (r, d) in zip(ref, deg))
                )
            )
            / length
        )
    else:
        return sum(_stoi(r, d, sr) for (r, d) in zip(ref, deg)) / length
