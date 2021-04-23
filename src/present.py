from librosa import display as librosa_display
from matplotlib import pyplot as plt
from scipy.io import wavfile


def wavfile_multi_write(filename, rate, data):
    for (i, d) in enumerate(data, start=1):
        wavfile.write(filename=f"{i}_{filename}", rate=rate, data=d)


def save_multi_wav(data, fname):
    for (i, d) in enumerate(data, start=1):
        save_wav(data=d, fname=f"{i}_{fname}")


def save_multi_spec(data, fname):
    for (i, d) in enumerate(data, start=1):
        save_spec(data=d, fname=f"{i}_{fname}")


def save_wav(data, fname, sr=16000):
    plt.clf()
    librosa_display.waveplot(y=data, sr=sr)
    plt.savefig(fname=fname)


def save_spec(data, fname, sr=16000):
    plt.clf()
    librosa_display.specshow(data=data, sr=sr)
    plt.savefig(fname=fname)
