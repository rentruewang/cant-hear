import functools
import itertools
import json
import os
from collections import defaultdict
from os import path as os_path

import librosa
import numpy as np
from librosa import effects
from librosa import feature as librosa_feature
from librosa import util as librosa_util
from matplotlib import pyplot as plt
from numpy import random as np_random
from torch import nn
from tqdm import tqdm


@functools.wraps(tqdm)
def progbar(iterable, total: int, message: str) -> tqdm:
    print(message)
    return tqdm(iterable=iterable, total=total)


def terminate_on_nan(_: nn.Module, grad_input: tuple, grad_output: tuple) -> bool:
    "Terminates the training process if NaN is found."

    nan_in = ((g is None or torch.isnan(g)) for g in grad_input)
    nan_out = ((g is None or torch.isnan(g)) for g in grad_output)
    if any(nan_in) or any(nan_out):
        print("NaN encountered")
        raise SystemExit


def name(path: str, fname: str, *args: tuple) -> str:
    "Generate the name for a file"
    return os_path.join(path, "_".join(fname.replace(".", "/").split("/") + list(args)))


def read_wav(
    fname: str, sr: int, norm: float = 0, pre_emphasis: bool = False
) -> np.ndarray:
    "Read a wave file into a normalized array"

    (S, _) = librosa.load(fname, sr=sr)
    (S, _) = effects.trim(S)
    if pre_emphasis:
        S[1:] -= S[:-1]
    if norm is not 0:
        S = librosa_util.normalize(S, norm=norm)
    return S


def stft(
    S: np.ndarray, n_fft: int, hop_length: int, win_length: int, window: str, norm: int
) -> np.ndarray:
    "Short time Fourier transform"

    S_hat = librosa.stft(
        y=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )
    S_hat = (abs(S_hat) ** norm).astype("float32")
    return S_hat


def melspectrogram(
    arr: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    power: int,
):
    "mel spectrogram of the original sequence"

    return librosa_feature.melspectrogram(
        y=arr,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        power=power,
    )


def power(seq: np.ndarray) -> np.ndarray:
    "Power of the original voice sequence"

    return (seq ** 2).sum()


def rescale(input: np.ndarray, target: np.ndarray, ratio: float) -> np.ndarray:
    "Rescale the input"

    return input * np.sqrt((power(target) / power(input + 1e-8)) / ratio)


def bernoulli(length: int, prob: float) -> np.ndarray:
    "Bernoulli distribution"

    rand = np_random.uniform(low=0, high=1, size=[length])
    return (rand < prob).astype("int")


def softmax(x: np.ndarray, dim: int) -> np.ndarray:
    "Perform softmax"

    exp = np.exp(x)
    return exp / exp.sum(axis=dim)


def select_indices(length: int, prob: float) -> np.ndarray:
    "Select several random indices"

    return np.argwhere(bernoulli(length, prob)).squeeze(-1)


def crop1d(array: np.ndarray, length: int) -> np.ndarray:
    "Cut the input into desired shape"

    assert array.ndim == 1
    if len(array) > length:
        index = np_random.randint(low=0, high=len(array) - length, size=None)
        return array[index : index + length]
    else:
        return array


def pad1d(array: np.ndarray, pads: tuple) -> np.ndarray:
    "Pad the input"

    assert len(pads) == 2
    (l, r) = pads
    return np.concatenate((np.zeros([l]), array, np.zeros([r])))


def mixing1d(arrays, indices, weights, target_length):
    "Mix different voices together"

    zeros = np.zeros(shape=[target_length])
    for (arr, ind, w) in zip(arrays, indices, weights):
        zeros += w * pad1d(arr, (ind, target_length - len(arr) - ind))
    return zeros


def generate_noises(
    noises: np.ndarray, indices: np.ndarray, target_length: int, T: float
) -> np.ndarray:
    "Generate fake noises"

    selected = tuple(noises[i] for i in indices)
    lengths = np.array(tuple(len(t) for t in selected))
    starting_indices = np_random.randint(low=0, high=target_length, size=len(indices))
    ending_indices = starting_indices + lengths
    ending_indices[ending_indices > target_length] = target_length
    lengths = ending_indices - starting_indices
    selected = tuple(crop1d(a, l) for a, l in zip(selected, lengths))
    weights = softmax(np_random.randn(len(indices)) / T, 0)
    return mixing1d(selected, starting_indices, weights, target_length)


def unfold_dict(dictionary: dict) -> dict:
    "Flatten a dictionary"

    lst = []

    def unfold_dict_recursive(dictionary, prefix=""):
        "Flatten a dictionary recursively"

        for (key, elem) in dictionary.items():
            assert isinstance(key, str)
            name = "/".join((prefix, key))

            if isinstance(elem, dict):
                unfold_dict_recursive(elem, prefix=name)
            else:
                lst.append((name.lstrip("/"), elem))

    unfold_dict_recursive(dictionary)

    return dict(lst)


def length(iterable):
    "Flatten an interable and count its length"

    i = 0

    def length_recursive(iterable):
        "Count the length of an iterable recursively"

        if isinstance(iterable, (list, tuple)):
            for t in iterable:
                length_recursive(t)
        else:
            nonlocal i
            i += 1

    length_recursive(iterable)
    return i


def remove_folder(folder: str):
    "Remove a folder"

    def remove_recursive(folder):
        "Remove a folder recursively"

        files = os.listdir(folder)
        for f in files:
            if os_path.isfile(f):
                os.remove(f)
            elif os_path.isdir(f):
                remove_recursive(os_path.join(folder, f))
                os.rmdir(folder)
            else:
                raise ValueError

    remove_recursive(folder=folder)
