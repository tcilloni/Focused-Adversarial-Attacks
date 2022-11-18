import torch
import numpy.typing as npt
import numpy as np
from typing import Tuple, Callable, Union
from models.ssd300.utils import generate_dboxes, Encoder


class DetectionsProducer():
    def __init__(self, model:torch.nn.Module, model_name:str = None, 
            label_score_box_fn:Callable = None, bbox_postprocess_fn:Callable = None) -> None:
        '''
        Callable object to produce detections in format <labels, scores, bboxes>.
        The constructor requires the model to use to produce detections, and either:
        1) a model name to use a pre-defined set of post-processing function
        2) a pair of functions to post-process the model's outputs

        In the latter case, the first function is used to get raw labels, scores, and
        bounding boxes for each detected image. These must be in a numpy array and have
        shape [n,4] if bounding boxes, [n] otherwise.
        The second function converts the bounding boxes in FiftyOne's format, which is
        <x1, y1, width, height>, in percentage values of the image's size. If the format
        of the bounding boxes returned by the first function is different, this function
        should take care of converting to 51 format. Otherwise, just specify a lambda
        function that returns whatever input it's given.

        Args:
            model (torch.Module): pytorch model to run predictions on
            model_name (str, optional): one of detr, retinanet, frcnn, ssd300. Defaults to None.
            label_score_box_fn (Callable, optional): to produce model's outputs. Defaults to None.
            bbox_postprocess_fn (Callable, optional): to post-process bboxes. Defaults to None.
        '''
        self.model = model
        self.bboxes_are_absolute = model_name in ['frcnn', 'retinanet']

        if label_score_box_fn and bbox_postprocess_fn:
            self.label_score_box = label_score_box_fn
            self.bbox_postprocess = bbox_postprocess_fn
        else:
            self.label_score_box = label_score_box_functions[model_name]
            self.bbox_postprocess = bbox_postprocessing_functions[model_name]


    def __call__(self, data: Union[torch.Tensor, Tuple[torch.Tensor, int, int]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Generate postprocessed predictions from detector model.
        Given an image tensor or a tuple of <image tensor, <pad_height, pad_width>>,
        this callable method returns a triplet of <labels, scores, bboxes>.
        Each returned value is a torch tensor; the first is int, the latter are
        floats. To interpret this, the labels indicate the COCO class of the
        prediction; the score is how confident the model is on that prediction
        (in range 0-1), and the bounding boxes are relative to the image size in
        format <x,y,w,h>, where (x,y) is the top-left corner of the box and (w,h)
        are its dimensions.

        Args:
            data (Union[torch.Tensor, Tuple[torch.Tensor, int, int]]): \
                either a numpy image or a tuple of image and two ints to specify the padding
                
        Returns:
            torch.Tensor: labels in coco format of the detections
            torch.Tensor: confidence scores for the detections
            torch.Tensor: bounding boxes of the detections in 51 format
        '''
        # reformat data to include paddings if missing (retinanet special case)
        if type(data) not in [list, tuple]: data = (data, (0,0))

        labels, scores, bboxes = self.label_score_box(self.model, data[0])

        if self.bboxes_are_absolute:
            (_,_, h, w), (pad_h, pad_w) = data[0].shape, data[1]
            bboxes = absolute_to_relative_bboxes(bboxes, h, w, pad_h, pad_w)

        bboxes = self.bbox_postprocess(bboxes)

        return labels, scores, bboxes


'''
    Functions to produce labels, scores and boxes (unprocessed)
'''
ssd300_encoder = Encoder(generate_dboxes(model='ssd'))
ssd300_to_coco_idxs = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
retinanet_to_coco_idxs =  {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

def to_cpu(labels: torch.Tensor, scores: torch.Tensor, bboxes: torch.Tensor
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''
    Move 3 torch tensors from any device to CPU.
    TODO: refactor this to take any number of torch tensors.

    Args:
        labels (npt.NDArray[np.uint8]): labels tensor
        scores (npt.NDArray[np.float32]): scores tensor
        bboxes (npt.NDArray[np.float32]): bounding boxes tensor

    Returns:
        npt.NDArray[np.uint8]: labels numpy array
        npt.NDArray[np.float32]: scores numpy array
        npt.NDArray[np.float32]: bounding boxes numpy array
    '''
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    return labels, scores, bboxes

def frcnn_label_score_box(model: torch.nn.Module, image: torch.Tensor
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''
    Generate a prediction from torchvision's Faster R-CNN model.

    Args:
        model (torch.nn.Module): torch model in eval mode
        image (torch.Tensor): batched tensor image

    Returns:
        npt.NDArray[np.uint8]: labels numpy array
        npt.NDArray[np.float32]: scores numpy array
        npt.NDArray[np.float32]: bounding boxes numpy array
    '''
    with torch.no_grad():
        out = model(image)[0]
        return to_cpu(out['labels'], out['scores'], out['boxes'])

def detr_label_score_box(model: torch.nn.Module, image: torch.Tensor
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''
    Refer to ``frcnn_label_score_box`` for documentation.
    '''
    with torch.no_grad():
        out = model(image)
        logits = out['logits'].softmax(-1)[0,:,:-1]
        labels = logits.argmax(1)
        scores = torch.take_along_dim(logits, labels[:,None], 1)
        return to_cpu(labels, scores, out['pred_boxes'][0])

def retinanet_label_score_box(model: torch.nn.Module, image: torch.Tensor
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''
    Refer to ``frcnn_label_score_box`` for documentation.
    '''
    with torch.no_grad():
        out = model(image)
        out[1].cpu().apply_(retinanet_to_coco_idxs.get)
        return to_cpu(out[1], out[0], out[2])

def ssd300_label_score_box(model: torch.nn.Module, image: torch.Tensor
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''
    Refer to ``frcnn_label_score_box`` for documentation.
    '''
    with torch.no_grad():
        locs, probs = model(image)
        out = ssd300_encoder.decode_batch(locs, probs, nms_threshold=0.45)[0]
        out[1].cpu().apply_(ssd300_to_coco_idxs.get)
        return to_cpu(out[1], out[2], out[0])


''' 
    Bounding Boxes Post-processing 
'''
def absolute_to_relative_bboxes(bboxes: npt.NDArray, h: int, w: int,
    pad_h: int, pad_w: int) -> npt.NDArray:
    '''
    Convert absolute (integer) to relative bounding boxes.

    Args:
        bboxes (npt.NDArray): bounding boxes of shape [4, num]
        h (int): height of the image
        w (int): width of the image
        pad_h (int): padding on bottom of the image
        pad_w (int): padding on the right of the image

    Returns:
        npt.NDArray: relative (float) bounding boxes
    ''' 
    bboxes[:, [0,2]] /= (w - pad_w)
    bboxes[:, [1,3]] /= (h - pad_h)

    return bboxes

def cx_cy__to__x1_y1(bboxes: npt.NDArray) -> npt.NDArray:
    '''
    Convert bounding boxes from <cx, cy, w, h> to <x, y, w, h>.

    Args:
        bboxes (npt.NDArray): bounding boxes of shape [4, num]

    Returns:
        npt.NDArray: bounding boxes of shape [4, num]
    '''
    bboxes[:, 0] -= bboxes[:, 2] / 2
    bboxes[:, 1] -= bboxes[:, 3] / 2
    return bboxes

def x1_y1_x2_y2__to__x1_y1_w_h(bboxes: npt.NDArray) -> npt.NDArray:
    '''
    Convert bounding boxes from <x1, y1, x2, y2> to <x, y, w, h>.
    The first format is a two-point bounding box scheme, the latter
    uses the top-left corner and the box's width and height.

    Args:
        bboxes (npt.NDArray): bounding boxes of shape [4, num]

    Returns:
        npt.NDArray: bounding boxes of shape [4, num]
    '''
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    return bboxes


''' 
    For cleaner code 
'''
label_score_box_functions = {
    'frcnn':     frcnn_label_score_box,
    'retinanet': retinanet_label_score_box,
    'ssd300':    ssd300_label_score_box,
    'detr':      detr_label_score_box,
}

bbox_postprocessing_functions = {
    'frcnn':     x1_y1_x2_y2__to__x1_y1_w_h,
    'retinanet': x1_y1_x2_y2__to__x1_y1_w_h,
    'ssd300':    x1_y1_x2_y2__to__x1_y1_w_h,
    'detr':      cx_cy__to__x1_y1,
}