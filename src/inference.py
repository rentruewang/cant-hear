"""
This module is used when you want to evaluate the model you trained.
"""

import glob
import json
import logging
import os
import types
from argparse import ArgumentParser
from functools import wraps
from multiprocessing.pool import Pool
from os import path as os_path
from typing import Sequence, Union

import librosa
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch import cuda
from torch.nn import Module, ModuleList
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import present, utils
from .autoencoder import Model
from .datasets import PairedDataset
from .utils import MapWrapper


def _snr(reference: np.ndarray, tensor: np.ndarray) -> float:
    "Signal to noise ratio"
    assert reference.shape == tensor.shape, (reference.shape, tensor.shape)
    noise_pow = ((tensor - reference) ** 2).sum()
    ref_pow = (reference ** 2).sum()
    return ref_pow / noise_pow


def _prepare_model(model_config: dict, device: str) -> Module:
    "Load model from disk according to config"
    state_dict = model_config["state"]
    len_model = model_config["len"]
    connection = model_config["connection"]

    model = ModuleList(Model(connection=connection) for _ in range(len_model))
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    return model


def spec_wav(data: np.ndarray or tuple, fname: str, configs: dict, dirs: tuple):
    "Write spectrogram to wavfiles"
    if isinstance(data, Sequence):
        for (idx, entry) in enumerate(data):
            spec_wav(entry, fname + f"_{idx:02d}", configs, dirs)
    else:
        (vis_dir, voc_dir) = dirs
        present.save_spec(data=data, fname=os_path.join(vis_dir, f"{fname}.png"))
        data = librosa.griffinlim(
            S=data,
            n_iter=configs["n_iter"],
            hop_length=configs["hop_length"],
            win_length=configs["win_length"],
            window=configs["window"],
        )
        present.save_wav(data=data, fname=os_path.join(voc_dir, f"{fname}.png"))
        wavfile.write(
            filename=os_path.join(voc_dir, f"{fname}.wav"),
            rate=configs["sr"],
            data=data,
        )


def to_numpy(tensor_list):
    "Convert a torch tensor to an numpy array"
    if isinstance(tensor_list, torch.Tensor):
        return tensor_list.cpu().detach().numpy()
    return tuple(to_numpy(elem) for elem in tensor_list)


def inference(
    data: tuple,
    model_config: dict,
    device: str,
    target_dir: str,
    configs: dict,
    processes: int = 1,
    metric_only: bool = False,
    griffinlim: types.FunctionType = None,
    waveform_metrics: list = [],
    logger: logging.Logger = None,
):
    "Perform inference with the model"
    ((clean_train, dirty_train), (clean_test, dirty_test)) = data
    assert (
        clean_train.shape == dirty_train.shape == clean_test.shape == dirty_test.shape
    )
    assert logger is not None

    in_list = (clean_train, dirty_train, clean_test, dirty_test)

    model = _prepare_model(model_config=model_config, device=device)

    out_list = []
    for i in in_list:
        _list = [i]
        for layer in model:
            i = layer(i)
            _list.append(i)
        out_list.append(_list)

    in_list = to_numpy(in_list)
    out_list = to_numpy(out_list)

    (clean_train, dirty_train, clean_test, dirty_test) = in_list
    (clean_train_out, dirty_train_out, clean_test_out, dirty_test_out) = out_list
    reference = (*(2 * (clean_train,)), *(2 * (clean_test,)))

    (l1_loss, l2_loss) = (
        lambda input, target: F.l1_loss(
            input=torch.tensor(input, device=device),
            target=torch.tensor(target, device=device),
        ).cpu(),
        lambda input, target: F.mse_loss(
            input=torch.tensor(input, device=device),
            target=torch.tensor(target, device=device),
        ).cpu(),
    )

    out_tags = ("clean_train", "dirty_train", "clean_test", "dirty_test")

    logger.info("applying snr")
    snr_list = tuple(
        tuple(_snr(reference=r, tensor=t).item() for t in entry)
        for (r, entry) in zip(reference, out_list)
    )
    json.dump(
        obj=dict(zip(out_tags, snr_list)),
        fp=open(os_path.join(target_dir, "snr.json"), mode="w+"),
        indent=4,
    )

    logger.info("applying l1")
    l1_list = (
        tuple(l1_loss(input=t, target=r).item() for t in entry)
        for (r, entry) in zip(reference, out_list)
    )
    json.dump(
        obj=dict(zip(out_tags, l1_list)),
        fp=open(os_path.join(target_dir, "l1.json"), mode="w+"),
        indent=4,
    )

    logger.info("applying l2")
    l2_list = (
        tuple(l2_loss(input=t, target=r).item() for t in entry)
        for (r, entry) in zip(reference, out_list)
    )
    json.dump(
        obj=dict(zip(out_tags, l2_list)),
        fp=open(os_path.join(target_dir, "l2.json"), mode="w+"),
        indent=4,
    )

    # waveform metrics
    if len(waveform_metrics) != 0:
        pool = Pool(processes=processes) if processes != 1 else MapWrapper()
        logger.info("transforming to waveform")
        wave_list = (tuple(griffinlim(o, pool) for o in out) for out in out_list)
        wave_ref = (griffinlim(r, pool) for r in reference)
        for (metric, func, default) in waveform_metrics:
            default = default or {}
            logger.info(f"applying {metric}")
            value_list = (
                tuple(func(ref=r, deg=t, pool=pool, **default) for t in entry)
                for (r, entry) in zip(wave_ref, wave_list)
            )
            json.dump(
                obj=dict(zip(out_tags, value_list)),
                fp=open(os_path.join(target_dir, f"{metric}.json"), mode="w+"),
                indent=4,
            )

    if metric_only:
        return

    vis_dir = os_path.join(target_dir, "vis")
    os.makedirs(name=vis_dir, exist_ok=True)

    voc_dir = os_path.join(target_dir, "voc")
    os.makedirs(name=voc_dir, exist_ok=True)

    print("saving spectrograms and waves")
    pool = Pool(processes=processes) if processes != 1 else MapWrapper()

    # Convert data into an iterator
    data_iter = zip(
        clean_train,
        zip(*clean_train_out),
        dirty_train,
        zip(*dirty_train_out),
        clean_test,
        zip(*clean_test_out),
        dirty_test,
        zip(*dirty_test_out),
    )

    for (i, packed_data) in utils.progbar(
        iterable=enumerate(data_iter),
        total=len(clean_train),
        message="Save spectrogram to wav files",
    ):

        (
            c_train,
            c_train_o,
            d_train,
            d_train_o,
            c_test,
            c_test_o,
            d_test,
            d_test_o,
        ) = packed_data

        name_list = (
            f"c_train_{i:02d}",
            f"c_train_o_{i:02d}",
            f"d_train_{i:02d}",
            f"d_train_o_{i:02d}",
            f"c_test_{i:02d}",
            f"c_test_o_{i:02d}",
            f"d_test_{i:02d}",
            f"d_test_o_{i:02d}",
        )
        save_list = (
            c_train,
            c_train_o,
            d_train,
            d_train_o,
            c_test,
            c_test_o,
            d_test,
            d_test_o,
        )
        pool.starmap(
            func=spec_wav,
            iterable=(
                (s, n, configs, (vis_dir, voc_dir))
                for (s, n) in zip(save_list, name_list)
            ),
        )


def clear_relative(path: str) -> str:
    "Remove './' pattern"
    if path.startswith("./"):
        path = path.lstrip("./")
    path.replace("/./", "/")
    return path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=os_path.join("configs", "major.json")
    )
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-P", "--processes", type=int, default=0)
    parser.add_argument("-n", "--samples", type=int, default=128)
    parser.add_argument("-i", "--inference", type=str)
    parser.add_argument("-t", "--target", type=str, default="out")
    parser.add_argument("-m", "--metric-only", action="store_true")

    external = parser.add_argument_group("external")
    external.add_argument("-pesq", "--pesq", action="store_true")
    external.add_argument("-stoi", "--stoi", action="store_true")
    flags = parser.parse_args()

    sns.set()

    torch.set_grad_enabled(False)

    configs = json.load(fp=open(file=flags.config, mode="r"))
    path = flags.path
    processes = flags.processes or os.cpu_count()
    (train_path, test_path) = os_path.join(path, "train"), os_path.join(path, "test")
    device = flags.device if cuda.is_available() else "cpu"
    samples = flags.samples
    sr = configs["sr"]
    n_iter = configs["n_iter"]
    hop_length = configs["hop_length"]
    win_length = configs["win_length"]
    window = configs["window"]
    time_steps = configs["time_steps"]
    target = flags.target
    metric_only = flags.metric_only
    use_pesq = flags.pesq
    use_stoi = flags.stoi

    external_functions = []
    if use_pesq:
        from external import pesq

        external_functions.append(("pesq", pesq.pesq, {"sr": sr}))
    if use_stoi:
        from external import stoi

        external_functions.append(("stoi", stoi.stoi, {"sr": sr}))
    trainset = PairedDataset(
        filepath=os_path.join(train_path, "*"),
        processes=processes,
        time_steps=time_steps,
    )
    testset = PairedDataset(
        filepath=os_path.join(test_path, "*"),
        processes=processes,
        time_steps=time_steps,
    )

    trainloader = DataLoader(dataset=trainset, batch_size=samples)
    trainsamples = tuple(t.to(device) for t in next(iter(trainloader)))

    testloader = DataLoader(dataset=testset, batch_size=samples)
    testsamples = tuple(t.to(device) for t in next(iter(testloader)))

    inference_list = json.load(fp=open(file=flags.inference, mode="r"))

    logger = logging.getLogger()

    @wraps(librosa.griffinlim)
    def griffinlim(S: np.ndarray, pool: Union[Pool, MapWrapper]) -> np.ndarray:
        if S.ndim == 2:
            return librosa.griffinlim(
                S=S,
                n_iter=n_iter,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
            )
        if S.ndim == 3:
            return np.stack(
                pool.starmap(
                    func=librosa.griffinlim,
                    iterable=((_S, n_iter, hop_length, win_length, window) for _S in S),
                )
            )

        raise ValueError("Unreachable")

    for to_infer in inference_list:
        # summary
        history = os_path.join(to_infer, "history.json")
        print(history)
        history_dict = json.load(fp=open(file=history, mode="r"))
        for (key, value) in history_dict.items():
            td = os_path.join(target, to_infer)
            os.makedirs(name=td, exist_ok=True)
            fname = "_".join(clear_relative(key).split("/"))
            plt.clf()
            plt.plot(value)
            plt.savefig(f"{os_path.join(td, fname)}.png")

        # modules
        models = glob.glob(os_path.join(to_infer, "*.pt"))
        for module in models:
            print(module)
            model_dict = torch.load(module, map_location="cpu")
            td = os_path.join(
                target, "_".join(clear_relative(module).replace(".pt", "").split("/"))
            )
            os.makedirs(name=td, exist_ok=True)
            inference(
                data=(trainsamples, testsamples),
                model_config=model_dict,
                device=device,
                target_dir=td,
                configs=configs,
                processes=processes,
                metric_only=metric_only,
                griffinlim=griffinlim,
                waveform_metrics=external_functions,
                logger=logger,
            )
