import functools

import torch
from numpy import random as np_random
from torch import autograd, nn
from torch.nn import functional as F
from torch.nn import utils as nn_utils


def apply(fn) -> object:
    @functools.wraps(fn)
    def function(obj):
        return fn(obj)

    return function


@apply(lambda cls: cls.apply)
class GradReverse(autograd.Function):
    "Reverse the gradient for upstream tensors"

    @staticmethod
    def forward(ctx, x: torch.Tensor, lamb: torch.Tensor = None) -> torch.Tensor:
        "Pass through"
        if lamb is None:
            lamb = x.new_ones(())
        ctx.save_for_backward(lamb)
        return x * lamb

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor) -> torch.Tensor:
        "Reverse pass through"
        (lamb,) = ctx.saved_tensors
        return (-lamb * grad_x, None)


@apply(lambda cls: cls.apply)
class NoGrad(autograd.Function):
    "Remove the gradient by detaching the graph"

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        "Pass through"
        return x

    @staticmethod
    def backward(ctx, *_: tuple):
        "Reverse pass through"
        return None


@autograd.no_grad()
def signal_noise_ratio(clean: torch.Tensor, output: torch.Tensor) -> tuple:
    "Signal to noise ratio"
    noise = output - clean
    power_clean = (clean ** 2).sum()
    power_noise = (noise ** 2).sum()
    return (power_clean, power_noise)


def reconstruct(
    model: tuple,
    data: torch.Tensor,
    recon_loss: nn.Module,
    train: bool,
    max_norm: float,
) -> tuple:
    "Reconstruct the original sound"
    return supervised(
        model=model,
        data=(data, data),
        recon_loss=recon_loss,
        train=train,
        max_norm=max_norm,
    )


def flow(
    model: tuple,
    data: torch.Tensor,
    recon_loss: nn.Module,
    cycle: bool,
    train: bool,
    max_norm: float,
) -> tuple:
    "Perform the entire flow"

    assert train or not max_norm
    (model, optimizer) = model

    input = data
    loss = 0
    _inp = _inp_cycle = input
    for module in model:
        input = module(input)
        loss += recon_loss(input=input, target=_inp)
        _inp = input.detach()
    out = input
    if cycle:
        loss += recon_loss(input=out, target=_inp_cycle)

    if train:
        optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
    return (loss, out)


def random_timeshift(tensor: torch.Tensor) -> torch.Tensor:
    "Random timeshift on tensor"

    # delay or advance
    size = tensor.size(-1)
    start = np_random.randint(low=-size, high=size, size=[])
    end = start + size
    start = max(start, 0)
    end = min(end, size)
    return F.pad(input=tensor[..., start:end], pad=(start, size - end))


def random_scale(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    "Creates a tensor of random scale"

    r = np_random.uniform(low=-1, high=1)
    factor = scale ** r
    return tensor * factor


def artifact(
    processed: torch.Tensor, original: torch.Tensor, dropout: float, scale: float
) -> torch.Tensor:
    "Creates artificial noise"

    noise = original - processed
    noise = F.dropout(noise, dropout)
    noise = random_timeshift(noise)
    noise = random_scale(noise, scale)
    return processed + noise


@autograd.enable_grad()
def ambient(
    model: tuple,
    amb: tuple,
    data: tuple,
    wgan_ratio: int,
    measure_info: tuple,
    penalty: float,
    train: bool,
    max_norm: float,
) -> tuple:
    "Ambient loss for wasserstein GAN"

    assert train or not max_norm
    (model, model_optim) = model
    (amb, amb_optim) = amb
    (_, dirty) = data
    (dropout, scale) = measure_info

    for _ in range(wgan_ratio):
        out = amb(dirty)
        out = GradReverse(out)
        out = out.mean()

        grads = autograd.grad(
            out, amb.parameters(), create_graph=True, allow_unused=True
        )
        grad_norm = torch.cat([g.view(-1) for g in grads if g is not None], dim=0).norm(
            2
        )
        grad_norm = ((grad_norm - 1) ** 2) * penalty
        total = out + grad_norm

        if train:
            amb_optim.zero_grad()
            total.backward()
            nn_utils.clip_grad_norm_(amb.parameters(), max_norm=max_norm)
            amb_optim.step()

    _inp = dirty
    x = dirty
    for module in model:
        x = module(x)

    x = GradReverse(x)
    _x = artifact(x, _inp, dropout, scale)
    out = amb(_x)
    out = out.mean()

    grads = autograd.grad(out, amb.parameters(), create_graph=True, allow_unused=True)
    grad_norm = torch.cat([g.view(-1) for g in grads if g is not None], dim=0).norm(2)
    grad_norm = ((grad_norm - 1) ** 2) * penalty
    total = out + grad_norm

    if train:
        model_optim.zero_grad()
        amb_optim.zero_grad()
        total.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        nn_utils.clip_grad_norm_(amb.parameters(), max_norm=max_norm)
        model_optim.step()
        amb_optim.step()

    return ((out, grad_norm), x)


def supervised(
    model: tuple, data: tuple, recon_loss: nn.Module, train: bool, max_norm: float
) -> tuple:
    "Supervised training"
    assert train or not max_norm
    (model, model_optim) = model
    (clean, dirty) = data

    x = dirty
    for module in model:
        x = module(x)
    loss = recon_loss(input=x, target=clean)

    if train:
        model_optim.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        model_optim.step()

    return (loss, x)


class ReconstructionLoss(nn.Module):
    "How good is the reconstruct"

    def __init__(self, mode: str = "l1", reduction: str = "mean"):
        super().__init__()
        mode = mode.strip().lower()
        self.loss = {
            "l1": nn.L1Loss(reduction=reduction),
            "l2": nn.MSELoss(reduction=reduction),
            "smooth": nn.SmoothL1Loss(reduction=reduction),
        }[mode]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        "Pass through"
        return self.loss(input=input, target=target)


class ConcatLoss(nn.Module):
    "Perform loss on concated tensors"

    def __init__(self, *losses: tuple):
        super().__init__()
        self.losses = losses

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        "Pass through"
        return sum(loss(input=input, target=target) for loss in self.losses)
