"""
Use this module to train a model.
"""

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from os import path as os_path
from typing import NamedTuple

import torch
from torch import cuda, no_grad
from torch.nn import Module, ModuleList
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from . import init, losses, utils
from .autoencoder import Model, PatchDiscriminator
from .datasets import PairedDataset
from .losses import ReconstructionLoss
from .utils import ScalarRecorder


def get_model(connection: str, **configs: dict) -> Module:
    "Create the model from configs"
    num_models = configs["num_models"]
    return ModuleList(Model(connection) for _ in range(num_models))


ModelOptim = NamedTuple("ModelOptim", [("model", Module), ("optim", Optimizer)])


def train(
    epochs: int,
    loaders: tuple,
    model_info: ModelOptim,
    amb_info: ModelOptim,
    wgan_info: tuple,
    measure_info: tuple,
    cycle: bool,
    recon_loss: Module,
    device: str,
    max_norm: float,
    summary: str,
    save_interval: int,
    flags: tuple,
):
    "Training the model"
    (do_flow, do_recon, do_ambient, do_supv) = flags
    (train_loader, test_loader) = loaders

    model = model_info.model
    model_optim = model_info.optim
    amb = amb_info.model
    amb_optim = amb_info.optim

    (wgan_ratio, grad_penalty) = wgan_info

    try:
        len_model = len(model)
    except TypeError:
        len_model = 1

    os.makedirs(name=summary, exist_ok=True)
    recorder = ScalarRecorder(summary=summary)

    for epo in range(1, 1 + epochs):
        mean_V = defaultdict(list)
        mean_G = defaultdict(list)
        mean_R = defaultdict(list)
        mean_S = defaultdict(list)
        mean_N = defaultdict(list)
        mean_D = defaultdict(list)
        mean_F = defaultdict(list)
        ratio = defaultdict(list)

        for (clean, dirty) in utils.progbar(
            iterable=train_loader, message=f"Epoch {epo:04d}/{epochs:04d}, training."
        ):
            (clean, dirty) = (clean.to(device), dirty.to(device))
            model.train()
            if do_flow:
                (F_loss, out) = losses.flow(
                    model=(model, model_optim),
                    data=dirty,
                    recon_loss=recon_loss,
                    cycle=cycle,
                    train=True,
                    max_norm=max_norm,
                )
                mean_F["train"].append(F_loss.item())

            if do_ambient:
                ((value, grad_norm), out) = losses.ambient(
                    model=(model, model_optim),
                    amb=(amb, amb_optim),
                    data=(clean, dirty),
                    wgan_ratio=wgan_ratio,
                    measure_info=measure_info,
                    penalty=grad_penalty,
                    train=True,
                    max_norm=max_norm,
                )
                mean_V["train"].append(value.item())
                mean_G["train"].append(grad_norm.item())

            if do_recon:
                (R_loss, out) = losses.reconstruct(
                    model=(model, model_optim),
                    data=clean,
                    recon_loss=recon_loss,
                    train=True,
                    max_norm=max_norm,
                )
                mean_R["train"].append(R_loss.item())

            if do_supv:
                (loss, out) = losses.supervised(
                    model=(model, model_optim),
                    data=(clean, dirty),
                    recon_loss=recon_loss,
                    train=True,
                    max_norm=max_norm,
                )
                mean_D["train"].append(loss.item())

            (s_power, n_power) = losses.signal_noise_ratio(clean, out)

            mean_S["train"].append(s_power.item())
            mean_N["train"].append(n_power.item())

        with no_grad():
            for (clean, dirty) in utils.progbar(
                iterable=test_loader, message=f"Epoch {epo:04d}/{epochs:04d}, testing."
            ):
                (clean, dirty) = (clean.to(device), dirty.to(device))
                model.eval()

                if do_flow:
                    (F_loss, out) = losses.flow(
                        model=(model, model_optim),
                        data=dirty,
                        recon_loss=recon_loss,
                        cycle=cycle,
                        train=False,
                        max_norm=None,
                    )
                    mean_F["test"].append(F_loss.item())

                if do_ambient:
                    ((value, grad_norm), out) = losses.ambient(
                        model=(model, model_optim),
                        amb=(amb, amb_optim),
                        data=(clean, dirty),
                        wgan_ratio=wgan_ratio,
                        measure_info=measure_info,
                        penalty=grad_penalty,
                        train=False,
                        max_norm=None,
                    )

                    mean_V["test"].append(value.item())
                    mean_G["test"].append(grad_norm.item())

                if do_recon:
                    (R_loss, out) = losses.reconstruct(
                        model=(model, model_optim),
                        data=clean,
                        recon_loss=recon_loss,
                        train=False,
                        max_norm=None,
                    )
                    mean_R["test"].append(R_loss.item())

                if do_supv:
                    (loss, out) = losses.supervised(
                        model=(model, model_optim),
                        data=(clean, dirty),
                        recon_loss=recon_loss,
                        train=False,
                        max_norm=None,
                    )
                    mean_D["test"].append(loss.item())

                (s_power, n_power) = losses.signal_noise_ratio(clean, out)
                mean_S["test"].append(s_power.item())
                mean_N["test"].append(n_power.item())

        if do_flow:
            for key in mean_F.keys():
                value = mean_F[key]
                mean_F[key] = sum(value) / len(value)
            recorder(tag="flow", value=mean_F)

        if do_amb:
            for key in mean_V.keys():
                value = mean_V[key]
                mean_V[key] = sum(value) / len(value)
            recorder(tag="ambient", value=mean_V)
            for key in mean_G.keys():
                value = mean_G[key]
                mean_G[key] = sum(value) / len(value)
            recorder(tag="gradnorm", value=mean_G)

        if do_recon:
            for key in mean_R.keys():
                value = mean_R[key]
                mean_R[key] = sum(value) / len(value)
            recorder(tag="recon_loss", value=mean_R)

        if do_supv:
            for key in mean_D.keys():
                value = mean_D[key]
                mean_D[key] = sum(value) / len(value)
            recorder(tag="supv_loss", value=mean_D)

        for (key_S, key_N) in zip(mean_S.keys(), mean_N.keys()):
            key = key_N
            assert key == key_S
            (val_N, val_S) = (mean_N[key], mean_S[key])
            ratio[key] = (sum(val_S) / len(val_S)) / (sum(val_N) / len(val_N))
        recorder(tag="ratio", value=ratio)

        if epo % save_interval == 0:
            torch.save(
                obj={
                    "len": len_model,
                    "connection": connection,
                    "state": model.state_dict(),
                },
                f=os_path.join(summary, f"{epo:04d}.pt"),
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-nm", "--num-models", type=int, default=1)
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-dv", "--device", type=str, default="cuda")
    parser.add_argument("-P", "--processes", type=int, default=os.cpu_count())
    parser.add_argument("-sum", "--summary", type=str, required=True)
    parser.add_argument("-cc", "--custom", type=str, default="")
    parser.add_argument("-cn", "--connection", type=str, default="normal")
    parser.add_argument("-fl", "--flow", action="store_true")
    parser.add_argument("-rec", "--reconstruction", action="store_true")
    parser.add_argument("-amb", "--ambient", action="store_true")
    parser.add_argument("-sup", "--supervised", action="store_true")
    parser.add_argument("-no-init", "--no-init", action="store_true")
    parser.add_argument("-tc", "--taco", action="store_true")
    parser.add_argument("-od", "--out-dir", type=str, default="runs")
    flags = parser.parse_args()

    num_models = flags.num_models
    device = flags.device if cuda.is_available() else "cpu"
    processes = flags.processes
    summary = flags.summary
    summary = os_path.join(flags.out_dir, summary)
    os.makedirs(summary, exist_ok=True)
    my_custom = os_path.join(summary, "major.json")
    custom = flags.custom or my_custom
    do_init = not flags.no_init
    identifiers = ("original_clean", "original_dirty")

    path = flags.path
    train_path = os_path.join(path, "train")
    test_path = os_path.join(path, "test")
    do_flow = flags.flow
    do_recon = flags.reconstruction
    do_amb = flags.ambient
    do_sup = flags.supervised
    connection = flags.connection

    if os_path.exists(path=custom):
        major_configs = json.load(fp=open(file=custom, mode="r"))
        json.dump(obj=major_configs, fp=open(file=my_custom, mode="w"), indent=4)
    else:
        major_configs = json.load(
            fp=open(file=os_path.join("configs", "major.json"), mode="r")
        )
        json.dump(obj=major_configs, fp=open(file=my_custom, mode="w"), indent=4)

    assert any((do_flow, do_recon, do_amb, do_sup))

    epochs = major_configs["epochs"]
    batch_size = major_configs["batch_size"]
    betas = major_configs["betas"]
    cycle = major_configs["cycle"]
    dropout = major_configs["dropout"]
    scale = major_configs["scale"]
    lr = major_configs["lr"]
    max_norm = major_configs["max_norm"]
    save_interval = major_configs["save_interval"]
    time_steps = major_configs["time_steps"]
    grad_penalty = major_configs["grad_penalty"]
    wgan_ratio = major_configs["wgan_ratio"]

    trainset = PairedDataset(
        filepath=os_path.join(train_path, "*"),
        processes=processes,
        time_steps=time_steps,
        identifiers=identifiers,
    )
    testset = PairedDataset(
        filepath=os_path.join(test_path, "*"),
        processes=processes,
        time_steps=time_steps,
        identifiers=identifiers,
    )

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    model = get_model(connection=connection, num_models=num_models).to(device)
    model_optim = Adam(model.parameters(), lr=lr, betas=betas)
    amb = PatchDiscriminator().to(device)
    amb_optim = Adam(model.parameters(), lr=lr, betas=betas)
    recon_loss = ReconstructionLoss()
    loaders = (train_loader, test_loader)
    model_info = ModelOptim(model, model_optim)
    amb_info = ModelOptim(amb, amb_optim)
    wgan_info = ModelOptim(wgan_ratio, grad_penalty)
    measure_info = (dropout, scale)

    if do_init:
        model.apply(init.init)

    train(
        epochs=epochs,
        loaders=loaders,
        model_info=model_info,
        amb_info=amb_info,
        wgan_info=wgan_info,
        measure_info=measure_info,
        cycle=cycle,
        recon_loss=recon_loss,
        device=device,
        max_norm=max_norm,
        summary=summary,
        save_interval=save_interval,
        flags=(do_flow, do_recon, do_amb, do_sup),
    )
