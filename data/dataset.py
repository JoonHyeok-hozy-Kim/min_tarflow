import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data(dataset_name: str, img_size: int):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    if dataset_name == 'imagenet64':
        data_path = os.environ.get("DATAP_PATH_IMAGENET64")
        train_data_path = os.path.join(data_path, "train")
        valid_data_path = os.path.join(data_path, "val")
    else:
        raise ValueError(f"Unexpected dataset : {dataset_name}.")
    
    
    ds_train = datasets.ImageFolder(root=train_data_path, transform=transform)
    # ds_valid = datasets.ImageFolder(root=valid_data_path, transform=transform)
    
    # return ds_train, ds_valid
    return ds_train


def get_fid_stats_dir(dataset_name: str, img_size: int):    
    data_base_dir = 'data/'
    if dataset_name == 'imagenet64':
        fid_stats_dir = os.path.join(data_base_dir, dataset_name)
        fid_stats_dir = os.path.join(fid_stats_dir, f'fid_stats.pth')
        return fid_stats_dir
    else:
        raise ValueError(f"Unexpected dataset : {dataset_name}.")
    