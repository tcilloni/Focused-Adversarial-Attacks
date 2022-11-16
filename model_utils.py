import torch
import numpy.typing as npt
import numpy as np
from typing import Tuple
from models.ssd300.utils import generate_dboxes, Encoder


class DetectionsProducer():
    def __init__(self, model, model_name, label_score_box_fn=None, bbox_postprocess_fn=None) -> None:
        self.model = model
        self.bboxes_are_absolute = model_name in ['frcnn', 'retinanet']

        if label_score_box_fn and bbox_postprocess_fn:
            self.label_score_box = label_score_box_fn
            self.bbox_postprocess = bbox_postprocess_fn
        else:
            self.label_score_box = label_score_box_functions[model_name]
            self.bbox_postprocess = bbox_postprocessing_functions[model_name]

    def __call__(self, data) \
        -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        labels, scores, bboxes = self.label_score_box(self.model, data)

        if self.bboxes_are_absolute:
            bboxes = absolute_to_relative_bboxes(bboxes)

        bboxes = self.bbox_postprocess(bboxes)

        return labels, scores, bboxes


'''
    Functions to produce labels, scores and boxes (unprocessed)
'''
ssd300_encoder = Encoder(generate_dboxes(model='ssd'))

def to_cpu(labels: npt.NDArray[np.uint8], scores: npt.NDArray[np.float32], bboxes: npt.NDArray[np.float32]) \
    -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    return labels, scores, bboxes

def frcnn_label_score_box(model, image):
    with torch.no_grad():
        out = model(image)[0]
        return to_cpu(out['labels'], out['scores'], out['boxes'])

def detr_label_score_box(model, image):
    with torch.no_grad():
        out = model(image)[0]
        logits = out['logits'].softmax(-1)[:,:-1]
        labels = logits.argmax(1)
        scores = torch.take_along_dim(logits, labels[:,None], 1)
        return to_cpu(labels, scores, out['pred_boxes'])

def retinanet_label_score_box(model, image):
    with torch.no_grad():
        out = model(image)
        return to_cpu(out[0], out[1], out[2])

def ssd300_label_score_box(model, image):
    with torch.no_grad():
        locs, probs = model(image)
        out = ssd300_encoder.decode_batch(locs, probs, nms_threshold=0.45)[0]
        return to_cpu(out[1], out[2], out[0])


''' 
    Bounding Boxes Post-processing 
'''
def absolute_to_relative_bboxes(bboxes, params):
    if type(params) not in [list, tuple]: params = (params, (0,0))
    (_, h, w), (pad_w, pad_h) = params[0].shape, params[1]
    
    bboxes[:, [0,2]] /= (w - pad_w)
    bboxes[:, [1,3]] /= (h - pad_h)

    return bboxes

def cx_cy__to__x1_y1(bboxes):
    bboxes[:, 0] -= bboxes[:, 2] / 2
    bboxes[:, 1] -= bboxes[:, 3] / 2
    return bboxes

def x1_y1_x2_y2__to__x1_y1_w_h(bboxes):
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    return bboxes


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