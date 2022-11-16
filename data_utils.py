import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from PIL import Image
from const import DEVICE


standardize_T = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

de_standardize_T = T.Compose([
    T.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
    T.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
])


# pre-defined transforms
read_transforms = {
    'frcnn': T.ToTensor(),
    'detr': T.Compose([T.Resize(800), T.ToTensor(), standardize_T]),
    'ssd300': T.Compose([T.Resize(320), T.ToTensor(), standardize_T]),
    'retinanet': T.Compose([T.ToTensor(), 
        T.Resize(size=608, max_size=1024), standardize_T]),
}

inverse_transforms = {
    'frcnn': lambda x : x,
    'detr': de_standardize_T,
    'retinanet': de_standardize_T,
    'ssd300': de_standardize_T
}


class ImageHandler():
    def __init__(self, model_name: str=None, transform: T=None, 
        inv_transform: T=None) -> None:
        '''
        Stateful image handler.


        Args:
            model_name (str, optional): _description_. Defaults to None.
            transform (T, optional): _description_. Defaults to None.
            inv_transform (T, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
        '''
        if not model_name and not (transform and inv_transform):
            raise Exception('Must specify a model name or a transformation')
        
        if model_name:
            self.transform = read_transforms[model_name]
            self.inv_transform = inverse_transforms[model_name]
            self.pad32 = model_name == 'retinanet'
        
        if transform:
            self.transform = transform
            self.inv_transform = inv_transform


    def load(self, fname: str) -> Image:
        image = read_image(fname)
        image = self.transform(image)
        _, _, self.w, self.h = image.shape

        # special case for retinanet
        if self.pad32:
            pad_w = 32 - self.w % 32
            pad_h = 32 - self.h % 32
            image = F.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)
        
        return image


    def save_from_torch(self, fname: str, image: torch.Tensor) -> None:
        # special case for retinanet (if [w, h] are different from the image)
        image = image[:, :, :self.w, :self.h]
        image = self.inv_transform(image)
        image = tensor_to_numpy(image, was_standardized=False)
        save_numpy_image(fname, image)


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


def PIL_image_to_tensor(image: Image, standardize: bool = True, size: int = None, 
    transform: T = None) -> torch.Tensor:
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

    return transform(image).unsqueeze(0).to(DEVICE)


def tensor_to_numpy(image: torch.Tensor, was_standardized: bool = True
    ) -> npt.NDArray[np.uint8]:
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

