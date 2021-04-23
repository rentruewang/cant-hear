import glob
import itertools
import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from os import path as os_path

import numpy as np
from numpy import random as np_random

from . import present, utils

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--major-configs", type=str, default="configs/major.json")
    parser.add_argument("--path-configs", type=str, default="configs/paths.json")
    parser.add_argument("-n", "--normal", type=float, default=None)
    parser.add_argument("-s", "--signal", type=float, default=None)
    parser.add_argument("-V", "--visual", action="store_true")
    parser.add_argument("-N", "--normalize", action="store_true")
    parser.add_argument("-p", "--power", type=int, default=2)
    parser.add_argument("-M", "--mel-spec", action="store_true")
    parser.add_argument("-T", "--test", action="store_true")
    parser.add_argument("-P", "--pre-emph", action="store_true")
    flags = parser.parse_args()

    major_configs = json.load(fp=open(file=flags.major_configs, mode="r"))
    path_configs = json.load(fp=open(file=flags.path_configs, mode="r"))

    (normal, signal) = flags.normal, flags.signal
    visual = flags.visual
    normalize = flags.normalize
    spec_norm = flags.power
    mel = flags.mel_spec
    test = flags.test
    pre_emph = flags.pre_emph
    if mel:
        assert flags.power == 2

    mode = []
    if normal is not None:
        mode.extend(["normal", f"{normal}"])
    if signal is not None:
        mode.extend(["signal", f"{signal}"])
    mode = "_".join(mode)
    if not normalize:
        mode += f"_power_{spec_norm}_NN"
    if pre_emph:
        mode += "_PEM"

    sr = major_configs["sr"]
    n_fft = major_configs["n_fft"]
    hop_length = major_configs["hop_length"]
    win_length = major_configs["win_length"]
    window = major_configs["window"]

    processes = major_configs["processes"]

    noise_sources = major_configs["noise_sources"]
    T = major_configs["temparature"]

    # training, inference
    if not test:
        folder = path_configs["folder"]
        sub_train = os_path.join(folder, mode, path_configs["sub_train"])
        os.makedirs(sub_train, exist_ok=True)
        sub_test = os_path.join(folder, mode, path_configs["sub_test"])
        os.makedirs(sub_test, exist_ok=True)
        sub_noise = os_path.join(folder, mode, path_configs["sub_noise"])
        os.makedirs(sub_noise, exist_ok=True)

        if visual:
            vis_dir = path_configs["visualize"]
            vis_sub_train = os_path.join(vis_dir, mode, path_configs["sub_train"])
            os.makedirs(vis_sub_train, exist_ok=True)
            vis_sub_test = os_path.join(vis_dir, mode, path_configs["sub_test"])
            os.makedirs(vis_sub_test, exist_ok=True)
            vis_sub_noise = os_path.join(vis_dir, mode, path_configs["sub_noise"])
            os.makedirs(vis_sub_noise, exist_ok=True)

    with Pool(processes=processes) as pool:
        train_regex = path_configs["train_regex"]
        test_regex = path_configs["test_regex"]
        noise_regex = path_configs["noise_regex"]

        if isinstance(train_regex, str):
            train_regex = [train_regex]
        if isinstance(test_regex, str):
            test_regex = [test_regex]
        if isinstance(noise_regex, str):
            noise_regex = [noise_regex]

        train_files = sum((glob.glob(r) for r in train_regex), [])
        test_files = sum((glob.glob(r) for r in test_regex), [])
        noise_files = sum((glob.glob(r) for r in noise_regex), [])

        assert all((train_files, test_files, noise_files)), "Permission Denied"

        norm = np.inf if normalize else 0
        td_clean_train = pool.starmap(
            utils.read_wav,
            utils.progbar(
                iterable=((f, sr, norm, pre_emph) for f in train_files),
                total=len(train_files),
                message="reading train files.",
            ),
        )
        td_clean_test = pool.starmap(
            utils.read_wav,
            utils.progbar(
                iterable=((f, sr, norm, pre_emph) for f in test_files),
                total=len(test_files),
                message="reading test files.",
            ),
        )
        td_noise = pool.starmap(
            utils.read_wav,
            utils.progbar(
                iterable=((f, sr, norm, pre_emph) for f in noise_files),
                total=len(noise_files),
                message="reading noise files.",
            ),
        )

        fd_clean_train = pool.starmap(
            utils.stft,
            utils.progbar(
                iterable=(
                    (S, n_fft, hop_length, win_length, window, spec_norm)
                    for S in td_clean_train
                ),
                total=len(td_clean_train),
                message="stft on clean training data.",
            ),
        )
        fd_clean_test = pool.starmap(
            utils.stft,
            utils.progbar(
                iterable=(
                    (S, n_fft, hop_length, win_length, window, spec_norm)
                    for S in td_clean_test
                ),
                total=len(td_clean_test),
                message="stft on clean testing data.",
            ),
        )
        fd_noise = pool.starmap(
            utils.stft,
            utils.progbar(
                iterable=(
                    (S, n_fft, hop_length, win_length, window, spec_norm)
                    for S in td_noise
                ),
                total=len(td_noise),
                message="stft on noise data.",
            ),
        )

        if not test:
            pool.starmap(
                np.save,
                utils.progbar(
                    iterable=(
                        (utils.name(sub_train, file, "original_clean"), arr)
                        for (file, arr) in zip(train_files, fd_clean_train)
                    ),
                    total=len(train_files),
                    message="saving training spectrogram.",
                ),
            )
            pool.starmap(
                np.save,
                utils.progbar(
                    iterable=(
                        (utils.name(sub_test, file, "original_clean"), arr)
                        for (file, arr) in zip(test_files, fd_clean_test)
                    ),
                    total=len(test_files),
                    message="saving testing spectrogram.",
                ),
            )
            pool.starmap(
                np.save,
                utils.progbar(
                    iterable=(
                        (utils.name(sub_noise, file, "original_clean"), arr)
                        for (file, arr) in zip(noise_files, fd_noise)
                    ),
                    total=len(noise_files),
                    message="saving noise spectrogram.",
                ),
            )

        if mel:
            mel_fd_clean_train = pool.starmap(
                utils.melspectrogram,
                utils.progbar(
                    iterable=(
                        (S, sr, n_fft, hop_length, win_length, window, spec_norm)
                        for S in td_clean_train
                    ),
                    total=len(td_clean_train),
                    message="mel on clean training data.",
                ),
            )
            mel_fd_clean_test = pool.starmap(
                utils.melspectrogram,
                utils.progbar(
                    iterable=(
                        (S, sr, n_fft, hop_length, win_length, window, spec_norm)
                        for S in td_clean_test
                    ),
                    total=len(td_clean_test),
                    message="mel on clean testing data.",
                ),
            )
            mel_fd_noise = pool.starmap(
                utils.melspectrogram,
                utils.progbar(
                    iterable=(
                        (S, sr, n_fft, hop_length, win_length, window, spec_norm)
                        for S in td_noise
                    ),
                    total=len(td_noise),
                    message="mel on noise data.",
                ),
            )

            if not test:
                pool.starmap(
                    np.save,
                    utils.progbar(
                        iterable=(
                            (utils.name(sub_train, file, "mel_clean"), arr)
                            for (file, arr) in zip(train_files, mel_fd_clean_train)
                        ),
                        total=len(train_files),
                        message="saving mel training melspectrogram.",
                    ),
                )
                pool.starmap(
                    np.save,
                    utils.progbar(
                        iterable=(
                            (utils.name(sub_test, file, "mel_clean"), arr)
                            for (file, arr) in zip(test_files, mel_fd_clean_test)
                        ),
                        total=len(test_files),
                        message="saving mel testing melspectrogram.",
                    ),
                )
                pool.starmap(
                    np.save,
                    utils.progbar(
                        iterable=(
                            (utils.name(sub_noise, file, "mel_clean"), arr)
                            for (file, arr) in zip(noise_files, mel_fd_noise)
                        ),
                        total=len(noise_files),
                        message="saving mel noise melspectrogram.",
                    ),
                )

        if not test:
            if visual:
                pool.starmap(
                    present.save_spec,
                    utils.progbar(
                        iterable=(
                            (data, utils.name(vis_sub_train, file, "clean.png"))
                            for (data, file) in zip(fd_clean_train, train_files)
                        ),
                        total=len(train_files),
                        message="drawing training spectrogram.",
                    ),
                )
                pool.starmap(
                    present.save_spec,
                    utils.progbar(
                        iterable=(
                            (data, utils.name(vis_sub_test, file, "clean.png"))
                            for (data, file) in zip(fd_clean_test, test_files)
                        ),
                        total=len(test_files),
                        message="drawing testing spectrogram.",
                    ),
                )
                pool.starmap(
                    present.save_spec,
                    utils.progbar(
                        iterable=(
                            (data, utils.name(vis_sub_noise, file, "clean.png"))
                            for (data, file) in zip(fd_noise, noise_files)
                        ),
                        total=len(noise_files),
                        message="drawing noise spectrogram.",
                    ),
                )
                if mel:
                    pool.starmap(
                        present.save_spec,
                        utils.progbar(
                            iterable=(
                                (data, utils.name(vis_sub_train, file, "mel_clean.png"))
                                for (data, file) in zip(mel_fd_clean_train, train_files)
                            ),
                            total=len(train_files),
                            message="drawing mel training spectrogram.",
                        ),
                    )
                    pool.starmap(
                        present.save_spec,
                        utils.progbar(
                            iterable=(
                                (data, utils.name(vis_sub_test, file, "mel_clean.png"))
                                for (data, file) in zip(mel_fd_clean_test, test_files)
                            ),
                            total=len(test_files),
                            message="drawing mel testing spectrogram.",
                        ),
                    )
                    pool.starmap(
                        present.save_spec,
                        utils.progbar(
                            iterable=(
                                (data, utils.name(vis_sub_noise, file, "mel_clean.png"))
                                for (data, file) in zip(mel_fd_noise, noise_files)
                            ),
                            total=len(noise_files),
                            message="drawing mel noise spectrogram.",
                        ),
                    )

        if normal is not None:
            td_dirty_train_N = tuple(
                utils.progbar(
                    iterable=(np_random.randn(*t.shape) for t in td_clean_train),
                    total=len(td_clean_train),
                    message="generating dirty training data.",
                )
            )
            td_dirty_test_N = tuple(
                utils.progbar(
                    iterable=(np_random.randn(*t.shape) for t in td_clean_test),
                    total=len(td_clean_test),
                    message="generating dirty testing data.",
                )
            )

            td_dirty_train_N = tuple(
                utils.progbar(
                    iterable=(
                        utils.rescale(d, c, 10 ** (normal / 20))
                        for (d, c) in zip(td_dirty_train_N, td_clean_train)
                    ),
                    total=len(td_dirty_train_N),
                    message="rescale on training noises.",
                )
            )
            td_dirty_test_N = tuple(
                utils.progbar(
                    iterable=(
                        utils.rescale(d, c, 10 ** (normal / 20))
                        for (d, c) in zip(td_dirty_test_N, td_clean_test)
                    ),
                    total=len(td_dirty_test_N),
                    message="rescale on testing noises.",
                )
            )
        else:
            td_dirty_train_N = td_dirty_test_N = itertools.repeat(0)

        if signal is not None:
            index_sample_train = tuple(
                utils.progbar(
                    iterable=(
                        utils.select_indices(len(td_noise), noise_sources)
                        for _ in range(len(td_clean_train))
                    ),
                    total=len(td_clean_train),
                    message="selecting training mixing indices.",
                )
            )
            index_sample_test = tuple(
                utils.progbar(
                    iterable=(
                        utils.select_indices(len(td_noise), noise_sources)
                        for _ in range(len(td_clean_test))
                    ),
                    total=len(td_clean_test),
                    message="selecting testing mixing indices.",
                )
            )

            td_dirty_train_S = tuple(
                utils.progbar(
                    iterable=(
                        utils.generate_noises(td_noise, idx, len(t), T)
                        for (idx, t) in zip(index_sample_train, td_clean_train)
                    ),
                    total=len(td_clean_train),
                    message="generating dirty training data.",
                )
            )
            td_dirty_test_S = tuple(
                utils.progbar(
                    iterable=(
                        utils.generate_noises(td_noise, idx, len(t), T)
                        for (idx, t) in zip(index_sample_test, td_clean_test)
                    ),
                    total=len(td_clean_test),
                    message="generating dirty testing data.",
                )
            )

            td_dirty_train_S = tuple(
                utils.progbar(
                    iterable=(
                        utils.rescale(d, c, 10 ** (signal / 20))
                        for (d, c) in zip(td_dirty_train_S, td_clean_train)
                    ),
                    total=len(td_dirty_train_S),
                    message="rescale on training noises.",
                )
            )
            td_dirty_test_S = tuple(
                utils.progbar(
                    iterable=(
                        utils.rescale(d, c, 10 ** (signal / 20))
                        for (d, c) in zip(td_dirty_test_S, td_clean_test)
                    ),
                    total=len(td_dirty_test_S),
                    message="rescale on testing noises.",
                )
            )
        else:
            td_dirty_train_S = td_dirty_test_S = itertools.repeat(0)

        td_dirty_train = tuple(
            utils.progbar(
                iterable=(n + s for (n, s) in zip(td_dirty_train_N, td_dirty_train_S)),
                total=len(td_clean_train),
                message="adding two training modes",
            )
        )
        td_dirty_test = tuple(
            utils.progbar(
                iterable=(n + s for (n, s) in zip(td_dirty_test_N, td_dirty_test_S)),
                total=len(td_clean_test),
                message="adding two testing modes",
            )
        )

        fd_dirty_train = pool.starmap(
            utils.stft,
            utils.progbar(
                iterable=(
                    (S, n_fft, hop_length, win_length, window, spec_norm)
                    for S in td_dirty_train
                ),
                total=len(td_dirty_train),
                message="stft on training noises.",
            ),
        )
        fd_dirty_test = pool.starmap(
            utils.stft,
            utils.progbar(
                iterable=(
                    (S, n_fft, hop_length, win_length, window, spec_norm)
                    for S in td_dirty_test
                ),
                total=len(td_dirty_test),
                message="stft on testing noises.",
            ),
        )

        train_data = tuple(
            utils.progbar(
                iterable=(c + d for (c, d) in zip(fd_clean_train, fd_dirty_train)),
                total=len(fd_dirty_train),
                message="adding training data.",
            )
        )
        test_data = tuple(
            utils.progbar(
                iterable=(c + d for (c, d) in zip(fd_clean_test, fd_dirty_test)),
                total=len(fd_dirty_test),
                message="adding testing data.",
            )
        )
        if mel:
            mel_train_data = pool.starmap(
                utils.melspectrogram,
                utils.progbar(
                    iterable=(
                        (c + d, sr, n_fft, hop_length, win_length, window, spec_norm)
                        for (c, d) in zip(td_clean_train, td_dirty_train)
                    ),
                    total=len(fd_dirty_train),
                    message="converting training to mel.",
                ),
            )
            mel_test_data = pool.starmap(
                utils.melspectrogram,
                utils.progbar(
                    iterable=(
                        (c + d, sr, n_fft, hop_length, win_length, window, spec_norm)
                        for (c, d) in zip(td_clean_test, td_dirty_test)
                    ),
                    total=len(fd_dirty_test),
                    message="converting testing to mel.",
                ),
            )

        if not test:
            pool.starmap(
                np.save,
                utils.progbar(
                    iterable=(
                        (utils.name(sub_train, file, "original_dirty"), arr)
                        for (file, arr) in zip(train_files, train_data)
                    ),
                    total=len(train_data),
                    message="saving training data.",
                ),
            )
            pool.starmap(
                np.save,
                utils.progbar(
                    iterable=(
                        (utils.name(sub_test, file, "original_dirty"), arr)
                        for (file, arr) in zip(test_files, test_data)
                    ),
                    total=len(test_data),
                    message="saving testing data.",
                ),
            )

            if mel:
                pool.starmap(
                    np.save,
                    utils.progbar(
                        iterable=(
                            (utils.name(sub_train, file, "mel_dirty"), arr)
                            for (file, arr) in zip(train_files, mel_train_data)
                        ),
                        total=len(train_data),
                        message="saving mel training data.",
                    ),
                )
                pool.starmap(
                    np.save,
                    utils.progbar(
                        iterable=(
                            (utils.name(sub_test, file, "mel_dirty"), arr)
                            for (file, arr) in zip(test_files, mel_test_data)
                        ),
                        total=len(test_data),
                        message="saving mel testing data.",
                    ),
                )
            if visual:
                pool.starmap(
                    present.save_spec,
                    utils.progbar(
                        iterable=(
                            (data, utils.name(vis_sub_train, file, "dirty.png"))
                            for (data, file) in zip(train_data, train_files)
                        ),
                        total=len(train_data),
                        message="drawing training data.",
                    ),
                )
                pool.starmap(
                    present.save_spec,
                    utils.progbar(
                        iterable=(
                            (data, utils.name(vis_sub_test, file, "dirty.png"))
                            for (data, file) in zip(test_data, test_files)
                        ),
                        total=len(test_data),
                        message="drawing testing data.",
                    ),
                )
                if mel:
                    pool.starmap(
                        present.save_spec,
                        utils.progbar(
                            iterable=(
                                (data, utils.name(vis_sub_train, file, "mel_dirty.png"))
                                for (data, file) in zip(mel_train_data, train_files)
                            ),
                            total=len(train_data),
                            message="drawing mel training data.",
                        ),
                    )
                    pool.starmap(
                        present.save_spec,
                        utils.progbar(
                            iterable=(
                                (data, utils.name(vis_sub_test, file, "mel_dirty.png"))
                                for (data, file) in zip(mel_test_data, test_files)
                            ),
                            total=len(test_data),
                            message="drawing mel testing data.",
                        ),
                    )
