import torch
import torchvision.transforms as T
import numpy as np
import numpy.typing as npt
from PIL import Image
import yaml


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

standardize_T = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

de_standardize_T = T.Compose([
    T.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
    T.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
])


# pre-defined transforms
read_transforms = {
    'frcnn': T.Compose([T.ToTensor()])
}


def read_image(fname: str) -> Image:
    '''
    Read image with PIL.

    Args:
        fname (str): relative or absolute image filepath

    Returns:
        Image: Image type image (can be printed in jupyter directly)
    '''
    return Image.open(fname).convert('RGB')


def save_numpy_image(fname: str, image: npt.NDArray[np.uint8]) -> None:
    '''
    Save a numpy image to file.

    Args:
        fname (str): relative or absolute image filepath
        image (np.array): numpy array of integer values
    '''
    Image.fromarray(image.transpose((1,2,0))).save(fname)


def PIL_image_to_tensor(image: Image, standardize: bool = True, size: int = None, transform: T = None) -> torch.Tensor:
    '''
    Convert PIL image to torch tensor.
    The image is returned normalized, or standardized if the flag is raised.
    Standardization is done with ImageNet's statistics.

    Args:
        image (Image): _description_
        standardize (bool): the image is standardized if `True
        size (int): resize the image to size

    Returns:
        torch.Tensor: tensor image of shape [1,3,h,w]
    '''
    if not transform:
        transforms = [T.ToTensor()]

        if size:
            transforms.append(T.Resize(size))

        if standardize:
            transforms.append(standardize_T)

        transform = T.Compose(transforms)

    return transform(image).unsqueeze(0).to(config['device'])


def tensor_to_numpy(image: torch.Tensor, was_standardized: bool = True) -> npt.NDArray[np.uint8]:
    '''
    Convert torch tensor back to numpy.

    Args:
        image (torch.Tensor): image to convert
        was_standardized (bool, optional): if `True` standardization is reverted. Defaults to True.

    Returns:
        npt.NDArray[np.uint8]: numpy array of the image
    '''
    if was_standardized:
        image = de_standardize_T(image)

    image = image.cpu().squeeze(0).numpy()
    image = np.clip(image, 0, 1)
    image = (image * 255.).astype(np.uint8)

    return image

