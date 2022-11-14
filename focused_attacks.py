import torch


quantiles = {0: (.5), 1: (.5 + .68/2), 2: (.5 + .95/2), 3: (.5 + .997/2)}


def find_threshold(activations: torch.tensor, standard_deviations: int = 1) -> float:
    '''
    Find the threshold that filters out all activations but those 
    ``standard_deviations`` greater than the mean.
    If one were to filter ``activations`` by ``[activations > t]``, it
    would result in the 50 / 16 / 2.5 / 0.015 % highest numbers, depending on
    0/1/2/3 ``standard_deviations``.

    Args:
        activations (torch.tensor): tensor of any shape
        standard_deviations (int, optional): number of STDs to filter out. Defaults to 1.

    Returns:
        float: threshold to filter the tensor
    '''
    assert standard_deviations in range(3+1), \
        f'Invalid standard deviations; expected in [0,1,2,3], found {standard_deviations}'

    q = quantiles[standard_deviations]
    return torch.quantile(activations.flatten(), q).item()


def l1_loss(y_pred: torch.tensor, threshold=None) -> torch.Tensor:
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
