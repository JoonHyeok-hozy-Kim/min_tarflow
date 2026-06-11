import math


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