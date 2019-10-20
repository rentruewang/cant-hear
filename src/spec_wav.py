import glob
import json
import os
from argparse import ArgumentParser
from os import path as os_path

import librosa
import numpy as np
import seaborn as sns
from numpy import random as np_random
from scipy.io import wavfile
from tqdm import tqdm

import present

if __name__ == "__main__":
    # the multiprocessing overhead isn't worth it
    parser = ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, nargs="+")
    parser.add_argument("-o", "--outdir", type=str, default="")
    flags = parser.parse_args()

    major_configs = json.load(fp=open("configs/major.json"))
    hop_length = major_configs["hop_length"]
    n_fft = major_configs["n_fft"]
    n_iter = major_configs["n_iter"]
    sr = major_configs["sr"]
    win_length = major_configs["win_length"]
    window = major_configs["window"]

    sns.set()

    infiles, outdir = flags.infile, flags.outdir

    os.makedirs(outdir, exist_ok=True)

    ifname = tuple(os_path.split(p=fp)[-1] for fp in infiles)
    specs = tuple(np.load(file=f) for f in infiles)

    for n, s in zip(ifname, specs):
        present.save_spec(data=s, fname=f"{n}-spec.png")

    for n, s in zip(ifname, specs):
        wav = librosa.griffinlim(
            S=s,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        wavfile.write(filename=f"{n}-wav.wav", rate=sr, data=wav)
        present.save_wav(data=wav, fname=f"{n}-time.png")
