"""
This module provides several datasets that are compatible with torch
to be used in torch's `DataLoader`s.
"""
import glob
from multiprocessing.pool import ThreadPool

import numpy as np
from numpy import random as np_random
from torch.utils.data import Dataset

from .utils import MapWrapper


def _dump_too_short(array: np.ndarray, target_len: int, dim: int = -1) -> np.ndarray:
    "Remove the input that is too short"
    return (array.shape[dim] > target_len) and array


def _select_slice(array: np.ndarray, max_len: int, dim: int = -1) -> np.ndarray:
    "Choose only a fixed length of the voice."
    randint = np_random.randint(low=0, high=array.shape[dim] - max_len)
    return np.stack(
        tuple(
            np.take(array, indices=i, axis=dim)
            for i in range(randint, randint + max_len)
        ),
        axis=dim,
    )


class PairedDataset(Dataset):
    "Pair the clean data with dirty data"

    def __init__(
        self,
        filepath: str,
        processes: int,
        time_steps: int,
        dim: int = -1,
        identifiers: tuple = ("original_clean", "original_dirty"),
        debug: bool = False,
    ):
        super().__init__()

        files = glob.glob(filepath)

        (clean, dirty) = identifiers
        clean_files = sorted(f for f in files if clean in f)
        dirty_files = sorted(f for f in files if dirty in f)
        assert len(clean_files) == len(dirty_files)

        if debug:
            _copy_clean = tuple(f.replace(clean, "") for f in clean_files)
            _copy_dirty = tuple(f.replace(dirty, "") for f in dirty_files)
            assert _copy_clean == _copy_dirty

        self.pool = ThreadPool(processes=processes) if processes != 1 else MapWrapper()
        clean_arrays = self.pool.map(np.load, clean_files)
        dirty_arrays = self.pool.map(np.load, dirty_files)
        assert len(clean_arrays) == len(dirty_arrays)

        clean_arrays = (_dump_too_short(arr, time_steps, dim) for arr in clean_arrays)
        clean_arrays = tuple(arr for arr in clean_arrays if bool(arr))
        dirty_arrays = (_dump_too_short(arr, time_steps, dim) for arr in dirty_arrays)
        dirty_arrays = tuple(arr for arr in dirty_arrays if bool(arr))
        assert len(clean_arrays) == len(dirty_arrays)

        self.clean_arrays = clean_arrays
        self.dirty_arrays = dirty_arrays

        self.len = len(clean_arrays)
        self.time_steps = time_steps
        self.dim = dim

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        (clean, dirty) = (self.clean_arrays[index], self.dirty_arrays[index])
        return self.pool.starmap(
            func=_select_slice,
            iterable=((clean, self.time_steps, self.dim), (dirty, self.time_steps)),
        )


class LogWrap(Dataset):
    "log(data)"

    def __init__(self, raw_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        items = self.raw_data[index]
        return tuple(np.log(spec) for spec in items)


class LogDataset(Dataset):
    "Pair data to log(data)"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.raw = PairedDataset(*args, **kwargs)
        self.log = LogWrap(self.raw)
        self.len = len(self.raw)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.raw[index], self.log[index])
