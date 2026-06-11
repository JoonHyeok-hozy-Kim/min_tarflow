import math
import os
from datetime import datetime

import torch
import matplotlib.pyplot as plt

def get_2d_dataset(n_points, dataset_name):
    if dataset_name == "spiral":
        return get_spiral_dataset(n_points)

def get_spiral_dataset(n_points, noise_std=0.5, turns=2.0):
    t = torch.rand(n_points).sqrt()
    max_theta = turns * 2 * math.pi
    theta = t * max_theta
    
    r = theta
    x = r * torch.cos(theta) + torch.randn(n_points) * noise_std
    y = r * torch.sin(theta) + torch.randn(n_points) * noise_std
    
    data = torch.stack([x, y], dim=1)   # (N, 2)
    data = data.unsqueeze(-1)           # (N, 2, 1)
    
    return data

def save_2d_dataset_image(data, img_size=8, save_dir=None, file_name=None):    
    plt.figure(figsize=(img_size, img_size))
    plt.scatter(data[:, 0], data[:, 1], c='#1f77b4', s=5, alpha=0.7) 
    plt.axis('equal') 
    plt.axis('off')
    # plt.title("Single Continuous Spiral Dataset")
    
    if save_dir is None:
        img_dir = os.path.join("results", "dataset")
        img_dir = os.path.join(img_dir, "get_spiral_dataset")
        os.makedirs(img_dir, exist_ok=True)
    else:
        assert os.path.exists(save_dir), f"save_dir(={save_dir}) does not exists."
        img_dir = save_dir
    
    if file_name is None:
        now_str = datetime.now().strftime("%y%m%d_%H%M")
        file_path = os.path.join(img_dir, f"two_dimensional-{now_str}.png")
    else:
        file_path = os.path.join(img_dir, f"{file_name}.png")
        
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
    n_points = 3000
    data = get_spiral_dataset(n_points)
    save_2d_dataset_image(data, n_points)