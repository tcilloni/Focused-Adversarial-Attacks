import torch
from models.ssd300.utils import generate_dboxes, Encoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SSD300_ENCODER = Encoder(generate_dboxes(model='ssd'))

FIFTYONE_EVAL_PARAMS = {
    'coco': {
        'pred_field':   'predictions', 
        'gt_field':     'detections', 
        'eval_key':     'eval',
        'method':       'coco',
        'compute_mAP':  True
    },
    'pascal': {
        'pred_field':   'predictions', 
        'gt_field':     'ground_truth', 
        'eval_key':     'eval',
        'method':       'open-images',
    }
}