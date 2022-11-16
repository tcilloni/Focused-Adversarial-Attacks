import os, shutil
import fiftyone as fo
import numpy.typing as npt
import numpy as np
from models.ssd300.utils import generate_dboxes, Encoder


coco_classes = {0: '0', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: '12', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: '26', 27: 'backpack', 28: 'umbrella', 29: '29', 30: '30', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: '45', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: '66', 67: 'dining table', 68: '68', 69: '69', 70: 'toilet', 71: '71', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: '83', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
pascal_classes = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane', 6: 'bus', 7: 'train', 9: 'boat', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 44: 'bottle', 62: 'chair', 63: 'sofa', 64: 'pottedplant', 67: 'diningtable', 72: 'tvmonitor', 73: 'tvmonitor'}
ssd300_encoder = Encoder(generate_dboxes(model='ssd'))


def fiftyone_coco_dataset(dir: str, use_cached: bool = False, max_samples: int = None):
    if dir in fo.list_datasets():
        if use_cached:
            return fo.load_dataset(dir)
        else:
            fo.delete_dataset(dir)
    
    coco_dataset = fo.Dataset.from_dir(
        dataset_type = fo.types.COCODetectionDataset,
        data_path = f'{dir}/images/val2017',
        labels_path = f'{dir}/images/labels.json',
        include_id = True,
        label_field = '',
        max_samples = max_samples,
        shuffle=False,
        name = dir
    )
    coco_dataset.persistent = True
    return coco_dataset


def fiftyone_pascal_dataset(dir: str, use_cached: bool = False, max_samples: int = None):
    if dir in fo.list_datasets():
        if use_cached:
            return fo.load_dataset(dir)
        else:
            fo.delete_dataset(dir)

    dataset = fo.Dataset.from_dir(
        dataset_dir = dir,
        dataset_type = fo.types.VOCDetectionDataset,
        name = dir,
        max_samples = max_samples
    )
    dataset.persistent = True

    return dataset


def prepare_coco_dataset_folder(src: str, dst: str):
    # make folder and copy fixed assets
    # os.makedirs(dst, exist_ok=True)
    shutil.copytree(f'{src}/annotations', f'{dst}/annotations', dirs_exist_ok=True)
    shutil.copytree(f'{src}/raw', f'{dst}/raw', dirs_exist_ok=True)
    shutil.copy(f'{src}/info.json', f'{dst}/info.json')

    # build data folder
    os.makedirs(f'{dst}/images/val2017', exist_ok=True)
    shutil.copy(f'{src}/images/labels.json', f'{dst}/images/labels.json')

    img_fnames = os.listdir(f'{src}/images/val2017')
    src_fnames = [f'{src}/images/val2017/{fname}' for fname in img_fnames]
    dst_fnames = [f'{dst}/images/val2017/{fname}' for fname in img_fnames]

    return src_fnames, dst_fnames


def prepare_pascal_dataset_folder(src: str, dst: str):
    # make folder and copy labels
    shutil.copytree(f'{src}/labels', f'{dst}/labels', dirs_exist_ok=True)

    # build data folder
    os.makedirs(f'{dst}/data', exist_ok=True)

    img_fnames = os.listdir(f'{src}/data')
    src_fnames = [f'{src}/data/{fname}' for fname in img_fnames]
    dst_fnames = [f'{dst}/data/{fname}' for fname in img_fnames]

    return src_fnames, dst_fnames


def produce_fiftyone_detection(dataset: str, labels: npt.NDArray[np.uint8],
        scores: npt.NDArray[np.float32],  bboxes: npt.NDArray[np.float32]) -> fo.Detections:
    classes = coco_classes if dataset == 'coco' else pascal_classes
    detections = []

    if len(scores) == 0:
        return fo.Detections(detections=[])

    # build detections list
    for label, score, bbox in zip(labels, scores, bboxes):
        if label in classes:
            detection = fo.Detection(label=classes[label], bounding_box=bbox.tolist(), confidence=float(score))
            detections.append(detection)
    
    return fo.Detections(detections=detections)

