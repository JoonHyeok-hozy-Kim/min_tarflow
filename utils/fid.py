import os
import datetime

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data.dataset import get_data
    

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