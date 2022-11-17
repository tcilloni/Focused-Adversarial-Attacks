import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        'compute_mAP':  True
    }
}