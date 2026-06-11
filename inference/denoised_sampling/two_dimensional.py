import argparse
import os
from datetime import datetime as dt

import torch

from models.tarflow import TarFlowRaw
from utils.seed import seed_everything
from data.two_dimensional import get_2d_dataset, save_2d_dataset_image


RANDOM_SEED = 46


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--pre_trained_weight_path', type=str, help='Pre-trained weight path')
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
    
    parser.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    parser.add_argument('--num_samples', default=4096, type=int, help='Number of sampels to draw')
    
    args = parser.parse_args()
    
    file_name = os.path.basename(__file__).split(".")[0]   
    now_str = dt.now().strftime("%y%m%d_%H%M")       
    results_dir = os.path.join(args.output_dir, f"inference/denoised_sampling/{file_name}")
    results_dir = os.path.join(results_dir, f"run-{now_str}")
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(RANDOM_SEED)
    
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
    
    assert os.path.exists(args.pre_trained_weight_path), f"pre_trained_weight_path(={args.pre_trained_weight_path}) does not exist."
    assert args.pre_trained_weight_path.split(".")[-1] == "pth", f"pre_trained_weight_path is not a .pth file: {args.pre_trained_weight_path}"
    checkpoint = torch.load(args.pre_trained_weight_path, weights_only=False)
    
    print("Loading state_dicts")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"=> Loaded model weights.")
    
    # Get Denoised Samples
    for param in model.parameters():
        param.requires_grad = False
    
    print("Sample from the model.")
    fixed_noise = torch.randn(args.num_samples, num_patches, 1).to(device)
    sample = model.reverse(fixed_noise, None, args.cfg_weight, args.attn_temp, args.annealed_guidance)
    x = sample.clone().detach()
    sample = sample.detach().cpu()
    save_2d_dataset_image(sample, args.img_size, results_dir, f"original_sample")    
    
    print("Calculate the gradient.")
    x.requires_grad = True
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        z, outputs, log_dets = model(x, None)
        loss = model.get_loss(z, log_dets)
    grad = torch.autograd.grad(loss, [x])[0]
    
    print("Desnoised sampling.")
    x.data = x.data - args.lr * grad
    x = x.detach().cpu()
    save_2d_dataset_image(x, args.img_size, results_dir, f"denoised_sample")
    
    print("fin.")