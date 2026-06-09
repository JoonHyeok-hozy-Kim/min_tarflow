import random
import math
import os
import glob
import datetime

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data.dataset import get_data

def seed_everything(seed_val: int):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    

def get_best_checkpoint(weight_dir):
    assert os.path.exists(weight_dir), f"Invalid path : {weight_dir}"
    final_format = weight_dir + "/model_final.pth"
    model_final_file = glob.glob(final_format)
    assert not model_final_file, f"Final already exists at {weight_dir}"
    
    best_format = weight_dir + "/model_best.pth"
    model_best_file = glob.glob(best_format)
    assert model_best_file, f"Best model not found at {weight_dir}"
    
    return model_best_file[0]


def get_wandb_run_info(run_url):
    # Get Info
    elements = run_url.split("//")[1].split("/")
    user_name = elements[1]
    project_name = elements[2]
    run_id = elements[4].split("?")[0]
    
    return user_name, project_name, run_id    

    
def get_lr_schedule(epoch, total_epochs, base_lr, lr_schedule_type, warm_up=False, start_epoch=None):
    if lr_schedule_type == 'sd':
        return get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up=False)
    elif lr_schedule_type == 'wsd':
        return get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up=True)
    elif lr_schedule_type == 'd':
        return get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up=False, decay_only=True)
    elif lr_schedule_type == 'ws':
        return get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up=True, decay_only=False, no_decay=True)
    elif lr_schedule_type == 's':
        return get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up=False, decay_only=False, no_decay=True)
    elif lr_schedule_type == 'cos':
        return get_cosine_lr_schedule(epoch, total_epochs, base_lr, warm_up=True)
    else:
        raise ValueError(f"Undefined lr schedule_type : {lr_schedule_type}")

def get_wsd_lr_schedule(epoch, total_epochs, base_lr, start_epoch, warm_up, decay_only=False, no_decay=False):
    warm_up_ratio = 0.05
    warm_up_start_lr_ratio = 0.1
    pivot_ratio = 0.5  
    min_lr_ratio = 0.01 
    
    actual_warm_up_end_epoch = min(total_epochs * warm_up_ratio, 10000)
    pivot_epoch = total_epochs * pivot_ratio
    
    if decay_only:
        if start_epoch is None:
            raise ValueError("start_epoch must be provided when decay_only is True")
        decay_progress = (epoch - start_epoch) / (total_epochs - start_epoch)
        lr_mult = 1.0 - (1.0 - min_lr_ratio) * decay_progress
    elif warm_up and epoch < actual_warm_up_end_epoch:
        alpha = epoch / actual_warm_up_end_epoch
        lr_mult = warm_up_start_lr_ratio + (1.0 - warm_up_start_lr_ratio) * alpha
    elif epoch < pivot_epoch or no_decay:
        lr_mult = 1.0
    else:
        decay_progress = (epoch - pivot_epoch) / (total_epochs - pivot_epoch)
        lr_mult = 1.0 - (1.0 - min_lr_ratio) * decay_progress
        
    return base_lr * lr_mult

def get_cosine_lr_schedule(epoch, total_epochs, base_lr, warm_up=True):
    warm_up_ratio = 0.05
    warm_up_start_lr_ratio = 0.1
    min_lr_ratio = 0.01  
    
    progress = epoch / total_epochs
    
    # 1. Warm-up 
    if warm_up and progress < warm_up_ratio:
        alpha = progress / warm_up_ratio
        lr_mult = warm_up_start_lr_ratio + (1.0 - warm_up_start_lr_ratio) * alpha
        return base_lr * lr_mult
    
    # 2. Cosine Annealing 
    if warm_up:
        adj_progress = (progress - warm_up_ratio) / (1.0 - warm_up_ratio)
    else:
        adj_progress = progress
        
    cos_out = 0.5 * (1.0 + math.cos(math.pi * adj_progress))
    lr_mult = min_lr_ratio + (1.0 - min_lr_ratio) * cos_out
    
    return base_lr * lr_mult


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


class FID(FrechetInceptionDistance):
    def add_state(self, name, default, *args, **kwargs):
        self.register_buffer(name, default)


def prepare_fid_stats(data, dataset_name, img_size, batch_size, fid_stat_dir):
    assert not os.path.exists(fid_stat_dir), f"File already exist : {fid_stat_dir}"
    print(f"[Prep.] Calculating FID scores for the first time.")
    fid = FID(reset_real_features=False, normalize=True).cuda()

    data_sampler = torch.utils.data.SequentialSampler(data)
    data_loader = torch.utils.data.DataLoader(
        data, sampler=data_sampler, batch_size=batch_size, num_workers=8, drop_last=False
    )

    for i, (x, _) in enumerate(data_loader):
        x = x.cuda()
        fid.update(x, real=True)

    torch.save(fid.state_dict(), fid_stat_dir)
    print(f'Saved FID stats file : {fid_stat_dir}')


class Distributed:
    def __init__(self):
        if os.environ.get('MASTER_PORT'):  # When running with torchrun
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            torch.distributed.init_process_group('nccl', 'env://', timeout=datetime.timedelta(minutes=10))
        else:  # When running with python for debugging
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
        torch.cuda.set_device(self.local_rank)
        self.barrier()

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.cat(x_list)

    def __del__(self):
        if self.distributed:
            torch.distributed.destroy_process_group()