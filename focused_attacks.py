import torch
from const import DEVICE
import numpy.typing as npt
from typing import Callable
from scipy.stats import gamma, kstest


def fa(model: torch.nn.Module, image: torch.Tensor, steps: int, epsilon: float,
        forward_fn: Callable, threshold: float = 0, dynamic: bool = False,
        percentile: float = 0.95) -> torch.Tensor:
    '''
    Focused Attacks (FA) algorithm main function.
    This function cloaks an image and returns it.
    By default it requires a threshold parameter; however, one can alternatively
    set ``dynamic`` to true and use a percentile and distribution based approach
    to determine the best thresholding value.

    Args:
        model (torch.nn.Module): object detector
        image (torch.Tensor): original tensor image to cloak
        steps (int): algorithmic iterations
        epsilon (float): maximum cumulative L1 distrortion (adv budget)
        forward_fn (Callable): function to get the predictions from the passed model
        threshold (float, Optional): focusing threshold. Defaults to 0.
        percentile (float, Optional): percentage of activations to filter out. Defaults to 0.

    Returns:
        torch.Tensor: cloaked image
    '''
    if dynamic:
        with torch.no_grad():
            out = forward_fn(model, image)
            threshold = torch.quantile(out.flatten(), percentile).item()

    # activate gradients
    mask = torch.zeros_like(image, requires_grad=True, device=DEVICE)
    image.requires_grad = True
    eps = epsilon / steps

    for _ in range(steps):
        # compute gradients
        out = forward_fn(model, mask + image)
        loss = selective_l1_loss(out, threshold=threshold)
        model.zero_grad()
        loss.backward()

        # update the mask
        mask.data -= mask.grad.data.detach().sign() * eps

        # prepare for next iteration
        mask.grad.data.zero_()
        image.grad.data.zero_()

    return (image + mask).detach()


def l1_loss(y_pred: torch.tensor) -> torch.Tensor:
    '''
    Basic L1 loss with 0 targets.
    This method calls torch.nn.L1Loss() with parameters y_pred and 0
    for all targets.

    Args:
        y_pred (torch.tensor): predictions of shape [batch_size, classes, locations]

    Returns:
        torch.Tensor: tensor of the loss to call .backward() upon 
    '''
    return torch.nn.L1Loss()(y_pred, torch.zeros_like(y_pred, device=y_pred.device))


def selective_l1_loss(y_pred: torch.tensor, threshold: float = 0.5) -> torch.Tensor:
    '''
    L1 loss with 0 targets that focuses only on relevant outputs.
    The relevance of an output is defined by whether or not it surpasses the threshold.
    Important features are selected by multiplying all non-important outputs by 0.

    Args:
        y_pred (torch.tensor): output of the model
        threshold (float, optional): threshold to consider outputs important. Defaults to 0.5.

    Returns:
        torch.Tensor: tensor of the loss to call .backward() upon 
    '''
    # only consider locations where items are found
    interesting_idxs = y_pred >= threshold              # [batch_size, classes, locations]
    y_pred = y_pred * interesting_idxs                  # [batch_size, locations, classes] * [batch_size, locations, classes]

    return l1_loss(y_pred)


def selective_l1_loss_indexed(y_pred: torch.tensor, threshold: float = 0.5) -> torch.Tensor:
    '''
    L1 loss with 0 targets that focuses only on relevant outputs.
    The relevance of an output is defined by whether or not it surpasses the threshold.
    Important features are selected by indexing important values only.
    This runs faster than its non-indexed counterpart when the outputs are sparse.

    Args:
        y_pred (torch.tensor): output of the model
        threshold (float, optional): threshold to consider outputs important. Defaults to 0.5.

    Returns:
        torch.Tensor: tensor of the loss to call .backward() upon 
    '''
    # only consider locations where items are found
    interesting_idxs = torch.where(y_pred >= threshold)
    y_pred_reduced = y_pred[interesting_idxs]

    return l1_loss(y_pred_reduced)
