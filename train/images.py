import argparse
import os
import wandb
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tvu

from models.tarflow import TarFlow
import utils
from data.two_dimensional import get_spiral_dataset


RANDOM_SEED = 46


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--generative_model_path', default='weights/', type=str, help='Model weight directory')
    parser.add_argument('--output_dir', default='results/', type=str, help='Output directory')
    
    parser.add_argument('--dataset_name', default="imagenet64", choices=["imagenet64"], help="Dataset name")
    parser.add_argument('--img_size', default=64, type=int, help="Image size")
    parser.add_argument('--channel_size', default=3, type=int, help="Image channel size")
    
    parser.add_argument('--patch_size', default=4, type=int, help="Patch size")
    parser.add_argument('--num_flow_blocks', default=8, type=int, help="Number of FlowBlocks")
    parser.add_argument('--flow_block_dim', default=1024, type=int, help="Internal dim of FlowBlocks")
    parser.add_argument('--num_attn_layers', default=8, type=int, help="Number of Attention blocks per FlowBlocks")
    parser.add_argument('--perumtation_type', default="flip", choices=["flip", "shuffle"], help="Type of permutation for the NF")
    parser.add_argument('--attn_head_dim', default=64, type=int, help="Head dim of Attention blocks")
    parser.add_argument('--attn_temp', default=1.0, type=float, help='Attention temperature')
    parser.add_argument('--ffn_multiplier', default=4, type=int, help="Internal dim multiplier for FFN layer")
    parser.add_argument('--cfg_weight', default=0, type=float, help='Guidance weight for sampling, 0 is no guidance')

    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size across all devices')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    parser.add_argument('--lr_schedule_type', default='wsd', type=str, choices=["wsd", "wd", "d", "cos"], help='Learning rate schedule')
    parser.add_argument('--class_dropout_prob', default=0, type=float, help='Ratio for random label drop in conditional mode')
    parser.add_argument('--sample_freq', default=1, type=int, help='Frequency of sampling in terms of epochs')
    parser.add_argument('--num_samples', default=4096, type=int, help='Number of sampels to draw')
    parser.add_argument('--sample_batch_size', default=256, type=int, help='Batch size for drawing samples')
    parser.add_argument('--resume_wandb_url', default='', type=str, help='URL at wandb')
    
    parser.add_argument(
        '--compile', default=False, action=argparse.BooleanOptionalAction, help='Whether to use torch.compile, expect the first epoch to be slow when enabled'
    )
    
    args = parser.parse_args()
    assert args.num_samples >= args.sample_batch_size, f"args.num_samples={args.num_samples} less than args.sample_batch_size(={args.sample_batch_size})."
    
    file_name = os.path.basename(__file__).split(".")[0]   
    now_str = datetime.now().strftime("%y%m%d_%H%M")    
    exp_config_dict = {}
    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        exp_config_dict[k] = v
        print(f'{k:32s}: {v}')
    
    wandb_exp_name = f"TARFlow-{file_name}-{args.dataset_name}-img_size"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.seed_everything(RANDOM_SEED)
    
    # ds_train, ds_valid = get_data(args.dataset_name, args.img_size)
    ds_train = get_data(args.dataset_name, args.img_size)
    num_classes = len(ds_train.classes)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    model = TarFlow(
        in_channels = args.channel_size,
        img_size = args.img_size,
        patch_size = args.patch_size,
        num_flow_blocks = args.num_flow_blocks,
        flow_block_dim = args.flow_block_dim,
        num_attn_layers = args.num_attn_layers,
        perumtation_type = args.perumtation_type,
        attn_head_dim = args.attn_head_dim,
        attn_temp= args.attn_temp,
        ffn_multiplier = args.ffn_multiplier,
        num_classes = num_classes,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters}, {num_parameters / 1e6}M")
    # print(f"Layer Norm: {args.use_layer_norm}, RoPE 2D : {args.use_rope_2d}")
    
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    # Resume related settings
    start_epoch = 0
    min_train_loss = torch.inf
    min_valid_loss = torch.inf    
    
    # Directory settings
    base_weights_dir = os.path.join(args.generative_model_path, f"train_weights_wip/{file_name}")    
    base_weights_dir = os.path.join(base_weights_dir, args.dataset_name)
    base_results_dir = os.path.join(args.output_dir, f"train/{file_name}")
    base_results_dir = os.path.join(base_results_dir, args.dataset_name)
    
    
    if args.resume_wandb_url and args.resume_wandb_url.lower() != "false":  # In case of bash passing false
        user_name, project_name, run_id = utils.get_wandb_run_info(args.resume_wandb_url)
        wandb.init(
            project=project_name,
            entity=user_name, 
            id=run_id,
            resume="must",
        )
            
        base_weights_dir = os.path.join(base_weights_dir, run_id)
        base_results_dir = os.path.join(base_results_dir, run_id)
            
        print(f"Resume training.")
        
        ws_weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_ws")
        assert os.path.exists(ws_weights_dir), f"weights_dir {ws_weights_dir} does not exist for resuming."
        
        print(f"=> Getting checkpoints from: {ws_weights_dir}")
        last_checkpoint = utils.get_best_checkpoint(ws_weights_dir)
        checkpoint = torch.load(last_checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"=> Loaded model weights.")
        
        if 'min_train_loss' in checkpoint: 
            min_train_loss = checkpoint['min_train_loss']
        
        # if 'min_valid_loss' in checkpoint: 
        #     min_valid_loss = checkpoint['min_valid_loss']
        
        print(f"=> Resuming from epoch {start_epoch}")
            
    else:
        run_id = os.environ.get("WANDB_RUN_ID", wandb.util.generate_id())
        print(f"Initiate wandb with new run_id : {run_id}")
        
        wandb.init(
            project=wandb_exp_name, 
            name=f"{file_name}",
            id=run_id,
            resume="allow",
            config = exp_config_dict,
        )
        
        base_weights_dir = os.path.join(base_weights_dir, run_id)
        base_results_dir = os.path.join(base_results_dir, run_id)
    
    # Directory settings based on wandb run_id
    weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_{args.lr_schedule_type}")
    results_dir = os.path.join(base_results_dir, f"lr_schedule_type_{args.lr_schedule_type}")    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop settings
    save_weight_freq = args.epochs // 10
    patience_cnt = 0
    tolerance_cnt = 10
    
    fixed_noise = torch.randn(
        args.num_samples,
        (args.img_size // args.patch_size) ** 2,
        args.channel_size * args.patch_size**2,
    )
    fixed_y = torch.randint(num_classes, (args.num_samples,))
    
    fid = utils.FID(reset_real_features=False, normalize=True).to(device)
    fid_stat_dir = get_fid_stats_dir(args.dataset_name, args.img_size)
    if os.path.exists(fid_stat_dir):
        print(f'Loading FID stats from {fid_stat_dir}')
        fid.load_state_dict(torch.load(fid_stat_dir, map_location='cpu', weights_only=False))
    else:
        utils.prepare_fid_stats(ds_train, args.dataset_name, args.img_size, args.batch_size, fid_stat_dir)
    
    def compute_loss(x, y, device, model):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            z, outputs, log_dets = model(x, y)
            loss = model.get_loss(z, log_dets)
            return loss, (z, outputs, log_dets)
    
    print(f"[torch.compile] : ", end="")
    if args.compile:
        compute_loss = torch.compile(compute_loss, fullgraph=False, backend='inductor', mode='max-autotune')
    print(f"fin.")
    
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        curr_lr = utils.get_lr_schedule(epoch, start_epoch + args.epochs, args.lr, args.lr_schedule_type, start_epoch=start_epoch)
        utils.set_lr(optimizer, curr_lr)
    
        model.train()
        for i, (x, y) in enumerate(dl_train):
            # print(f"dl_train cnt : {i+1}")
            if i > 0:
                break
            x, y = x.to(device), y.to(device)
            eps = torch.randn_like(x) * 0.05    # std_dev=0.05 from the paper
            x = x + eps
            
            # Class dropout
            dropout_mask = (torch.rand(y.size(0), device=device) < args.class_dropout_prob).int()
            y = (1 - dropout_mask) * y - dropout_mask   # dropout y = -1
            
            optimizer.zero_grad()
            loss, (z, outputs, log_dets) = compute_loss(x, y, device, model)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": curr_lr,
                "grad_norm": grad_norm.item(),
                "epoch": epoch,
            })
            
            min_train_loss = min(min_train_loss, loss.item())
        
        # Evaluate performance
        if (epoch + 1) % args.sample_freq == 0:
            model.eval()
            for i in range(args.num_samples // args.sample_batch_size):
                b = args.sample_batch_size
                noise = fixed_noise[i * b : (i + 1) * b].to('cuda')
                y = None if fixed_y is None else fixed_y[i * b : (i + 1) * b].to('cuda')
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        samples, _ = model.reverse(noise, y)
                        # samples = model.reverse(noise, y, guidance=args.cfg)
                        assert isinstance(samples, torch.Tensor), f"samples is not a Tensor ({type(samples)} instead)"
                    fid.update(0.5 * (samples.clip(min=-1, max=1) + 1), real=False)

            fid_score = fid.compute().item()
            fid.reset()

            wandb.log({
                "fid": fid_score,
            })
            img_dir = os.path.join(results_dir, f'samples_{epoch+1:03d}.png')
            tvu.save_image(samples, img_dir, normalize=True, nrow=16)
    
    wandb.finish()
        
