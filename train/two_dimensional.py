import argparse
import os
import wandb
from datetime import datetime as dt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tvu

from models.tarflow import TarFlowRaw
from utils.seed import seed_everything
from utils.fid import *
from utils.wandb import get_best_checkpoint, get_wandb_run_info
from utils.lr import get_lr_schedule, set_lr
from data.two_dimensional import get_2d_dataset, save_2d_dataset_image


RANDOM_SEED = 46


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--generative_model_path', default='weights/', type=str, help='Model weight directory')
    parser.add_argument('--output_dir', default='results/', type=str, help='Output directory')
    
    parser.add_argument('--dataset_name', default="spiral", choices=["spiral"], help="Dataset name")
    parser.add_argument('--img_size', default=8, type=int, help="Output image size")
    parser.add_argument('--channel_size', default=1, type=int, help="Channel size")
    
    parser.add_argument('--num_flow_blocks', default=8, type=int, help="Number of FlowBlocks")
    parser.add_argument('--flow_block_dim', default=1024, type=int, help="Internal dim of FlowBlocks")
    parser.add_argument('--permutation_type', default="flip", choices=["flip", "shuffle"], help="Type of permutation for the NF")
    parser.add_argument('--num_attn_blocks', default=8, type=int, help="Number of Attention blocks per FlowBlocks")
    parser.add_argument('--attn_num_heads', default=64, type=int, help="Head dim of Attention blocks")
    parser.add_argument('--attn_head_dim', default=64, type=int, help="Head dim of Attention blocks")
    parser.add_argument('--attn_temp', default=1.0, type=float, help='Attention temperature')
    parser.add_argument('--ffn_expansion', default=4, type=int, help="Internal dim multiplier for FFN layer")
    parser.add_argument('--cfg_weight', default=0.0, type=float, help='Guidance weight for sampling, 0 is no guidance')
    parser.add_argument("--annealed_guidance", action="store_true", help="Apply the annealed guidance")
    
    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size across all devices')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    parser.add_argument('--lr_schedule_type', default='wsd', type=str, choices=["wsd", "wd", "d", "cos"], help='Learning rate schedule')
    # parser.add_argument('--class_dropout_prob', default=0, type=float, help='Ratio for random label drop in conditional mode')
    parser.add_argument('--sample_freq', default=1, type=int, help='Frequency of sampling in terms of epochs')
    parser.add_argument('--num_samples', default=4096, type=int, help='Number of sampels to draw')
    # parser.add_argument('--sample_batch_size', default=256, type=int, help='Batch size for drawing samples')
    parser.add_argument("--dry_run", action="store_true", help="Simple test run in local.")
    parser.add_argument('--resume_wandb_url', default='', type=str, help='URL at wandb')
    
    parser.add_argument(
        '--compile', default=False, action=argparse.BooleanOptionalAction, help='Whether to use torch.compile, expect the first epoch to be slow when enabled'
    )
    
    args = parser.parse_args()
    # assert args.num_samples >= args.sample_batch_size, f"args.num_samples={args.num_samples} less than args.sample_batch_size(={args.sample_batch_size})."
    
    file_name = os.path.basename(__file__).split(".")[0]   
    now_str = dt.now().strftime("%y%m%d_%H%M")    
    exp_config_dict = {}
    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        exp_config_dict[k] = v
        print(f'{k:32s}: {v}')
    
    wandb_exp_name = f"TARFlow-train-{file_name}-{args.dataset_name}-img_size"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(RANDOM_SEED)
    
    # 2-Dim Unconditional Case
    num_classes = 0          
    num_patches = 2
    
    model = TarFlowRaw(
        in_channels = args.channel_size,
        num_patches = num_patches,     # 2-dimensional
        num_flow_blocks = args.num_flow_blocks,
        flow_block_dim = args.flow_block_dim,
        attn_num_heads = args.attn_num_heads,
        num_attn_blocks = args.num_attn_blocks,
        permutation_type = args.permutation_type,
        attn_head_dim = args.attn_head_dim,
        # attn_temp= args.attn_temp,
        ffn_expansion = args.ffn_expansion,
        num_classes = num_classes,
        # class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters}, {num_parameters / 1e6}M")
    # print(f"Layer Norm: {args.use_layer_norm}, RoPE 2D : {args.use_rope_2d}")
    
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    # Resume related settings
    start_epoch = 0
    min_train_loss = torch.inf
    # min_valid_loss = torch.inf    
    
    # Directory settings
    base_weights_dir = os.path.join(args.generative_model_path, f"train_weights_wip/{file_name}")    
    base_weights_dir = os.path.join(base_weights_dir, args.dataset_name)
    base_results_dir = os.path.join(args.output_dir, f"train/{file_name}")
    base_results_dir = os.path.join(base_results_dir, args.dataset_name)
    
    if args.dry_run:
        base_weights_dir = os.path.join(base_weights_dir, f"dry_run-{now_str}")
        base_results_dir = os.path.join(base_results_dir, f"dry_run-{now_str}")
            
    else:
        if args.resume_wandb_url and args.resume_wandb_url.lower() != "false":  # In case of bash passing false
            user_name, project_name, run_id = get_wandb_run_info(args.resume_wandb_url)
            wandb.init(
                project=project_name,
                entity=user_name, 
                id=run_id,
                resume="must",
            )
                
            base_weights_dir = os.path.join(base_weights_dir, f"wandb-{now_str}-{run_id}")
            base_results_dir = os.path.join(base_results_dir, f"wandb-{now_str}-{run_id}")
                
            print(f"Resume training.")
            
            ws_weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_ws")
            assert os.path.exists(ws_weights_dir), f"weights_dir {ws_weights_dir} does not exist for resuming."
            
            print(f"=> Getting checkpoints from: {ws_weights_dir}")
            last_checkpoint = get_best_checkpoint(ws_weights_dir)
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
                
            base_weights_dir = os.path.join(base_weights_dir, f"wandb-{now_str}-{run_id}")
            base_results_dir = os.path.join(base_results_dir, f"wandb-{now_str}-{run_id}")
    
    # Directory settings based on wandb run_id
    weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_{args.lr_schedule_type}")
    results_dir = os.path.join(base_results_dir, f"lr_schedule_type_{args.lr_schedule_type}")    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop settings
    save_weight_freq = args.epochs // 10
    patience_cnt = 0
    tolerance_cnt = 50
    
    fixed_noise = torch.randn(
        args.num_samples,
        num_patches,
        1,
    )
    fixed_y = None
    # fixed_y = torch.randint(num_classes, (args.num_samples,))
        
    # fid = FID(reset_real_features=False, normalize=True).to(device)
    # fid_stat_dir = get_fid_stats_dir(args.dataset_name, args.img_size)
    # if os.path.exists(fid_stat_dir):
    #     print(f'Loading FID stats from {fid_stat_dir}')
    #     fid.load_state_dict(torch.load(fid_stat_dir, map_location='cpu', weights_only=False))
    # else:
    #     prepare_fid_stats(ds_train, args.dataset_name, args.img_size, args.batch_size, fid_stat_dir)
    
    def compute_loss(x, y, device, model):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            z, outputs, log_dets = model(x, y)
            loss = model.get_loss(z, log_dets)
            return loss, (z, outputs, log_dets)
    
    # print(f"[torch.compile] : ", end="")
    # if args.compile:
    #     compute_loss = torch.compile(compute_loss, fullgraph=False, backend='inductor', mode='max-autotune')
    # print(f"fin.")
    
    
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        curr_lr = get_lr_schedule(epoch, start_epoch + args.epochs, args.lr, args.lr_schedule_type, start_epoch=start_epoch)
        set_lr(optimizer, curr_lr)
    
        model.train()
        x = get_2d_dataset(n_points=args.batch_size, dataset_name=args.dataset_name)
        if (epoch + 1) % args.sample_freq == 0:
            save_2d_dataset_image(x, args.img_size, results_dir, f"training_data-epoch_{epoch+1}")
        
        x = x.to(device)
        eps = torch.randn_like(x) * 0.05    # N(0, 0.05^2) from Tarflow 2.4
        x = x + eps
        y = None    # Unconditional
        
        optimizer.zero_grad()
        loss, (z, outputs, log_dets) = compute_loss(x, y, device, model)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if not args.dry_run:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": curr_lr,
                "grad_norm": grad_norm.item(),
                "epoch": epoch,
            })        
        
        curr_loss = loss.item()
        if curr_loss > min_train_loss:
            patience_cnt += 1
            if patience_cnt == tolerance_cnt:
                print(f"Early Termination.")
                break
        else:
            patience_cnt = 0
            
        min_train_loss = min(min_train_loss, curr_loss)
        
        
        # Evaluate performance
        if (epoch + 1) % args.sample_freq == 0:
            model.eval()
            noise = fixed_noise.clone().to(device)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z = model.reverse(noise, None, args.cfg_weight, args.attn_temp, args.annealed_guidance)
                    assert isinstance(z, torch.Tensor), f"samples is not a Tensor ({type(z)} instead)"
            #         fid.update(0.5 * (samples.clip(min=-1, max=1) + 1), real=False)

            # fid_score = fid.compute().item()
            # fid.reset()

            if not args.dry_run:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_train_loss' : min_train_loss,
                }
                torch.save(checkpoint, os.path.join(weights_dir, f"epoch_{epoch+1}-loss_{min_train_loss:.2f}.pth"))
            
            z = z.cpu()
            save_2d_dataset_image(z, args.img_size, results_dir, f"inference-epoch_{epoch+1}")
        
    
    if not args.dry_run:
        wandb.finish()
        
    exit()
    
    
    # Get Denoised Samples
    for param in model.parameters():
        param.requires_grad = False
    
    noise = fixed_noise.clone().to(device)
    sample = model.reverse(noise, None, 0.0, args.cfg_weight, args.attn_temp, args.annealed_guidance)
    x = sample.clone().detach()
    x.requires_grad = True
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, (z, outputs, log_dets) = compute_loss(x, None, device, model)
    grad = torch.autograd.grad(loss, [x])[0]
    x.data = x.data - args.lr * grad
    save_2d_dataset_image(x, args.img_size, results_dir, f"denoised_sample")