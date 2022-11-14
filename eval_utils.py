import os, shutil
import fiftyone as fo

# load_coco_dataset_in_fiftyone
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



# def delete_all_cached_datasets():
#     for dataset_name in fo.list_datasets():
#         fo.delete_dataset(dataset_name)


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

