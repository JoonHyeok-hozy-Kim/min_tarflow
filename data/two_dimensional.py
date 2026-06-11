import torch
import math
import matplotlib.pyplot as plt

def get_spiral_dataset(n_points, noise_std=0.5, turns=4.0):
    t = torch.rand(n_points).sqrt()