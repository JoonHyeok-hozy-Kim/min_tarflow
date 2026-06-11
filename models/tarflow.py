
import torch
import torch.nn as nn

from models.attention import Transformer
from permutation import PermutationIdentity, PermutationFlip

class FlowBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        flow_block_dim, 
        num_patches, 
        permutation, 
        num_attn_blocks, 
        attn_num_heads, 
        attn_head_dim, 
        ffn_expansion,
        num_classes,
    ):
        super().__init__()
        self.hidden_dim = flow_block_dim
        self.output_dim = in_channels * 2   # NVP        
        self.permutation = permutation
        
        self.w_in = nn.Linear(in_channels, flow_block_dim)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, flow_block_dim) * 1e-2)
        if num_classes == 0:
            self.class_embed = None
        else:
            self.class_embed = nn.Parameter(torch.randn(num_classes, 1, flow_block_dim) * 1e-2)
        self.transformer_blocks = nn.ModuleList([
            Transformer(flow_block_dim, attn_num_heads, attn_head_dim, ffn_expansion) for _ in range(num_attn_blocks)
        ])
        self.w_out = nn.Linear(flow_block_dim, self.output_dim)
        self.w_out.weight.data.fill_(0.0)
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        
    def forward(self, x, y, attn_temperature=1.0):
        x = self.permutation(x)
        pos_embed_perm = self.permutation(self.pos_embed, dim=0)
        x_perm = x  # "=i"
        
        x = self.wo_in(x) + pos_embed_perm
        if self.class_embed is not None:
            if y is not None:
                # Negative label for CFG
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)  # (Y, 1, 1)
                    class_embed = (1-m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                    
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, self.attn_mask, attn_temperature, cache_key=None)
        x = self.w_out(x)
        
        # Implement "<i" for alpha and mu by shifting one-slot right.
        x = torch.cat([
            torch.zeros_like(x[:, :1]),     # (B, 1, ...) all in 0
            x[:, :-1],                      # (B, T-1, ...) exclude the last one
        ], dim=1)                           # (B, T, ...)   [0, x_0, x_1, ..., x_(T-2)]
        
        alpha, mu = x.chunk(2, dim=-1)      # "<i"
        scale = torch.exp(-alpha.float()).type(alpha.dtype)
        x = self.permutation((x_perm - mu) * scale, inverse=True)
        return x, -alpha.mean(dim=[1, 2])   # Return mean instead of sum (Normalize by T)
    

class TarFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size,
        patch_size,
        flow_block_dim,
        num_flow_blocks,        
        num_attn_blocks, 
        attn_num_heads, 
        attn_head_dim, 
        ffn_expansion,
        num_classes,
    ):
        assert img_size % patch_size == 0, f"img_size(={img_size}) is not divided by patch_size(={patch_size})."
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.permutations = [PermutationIdentity(), PermutationFlip()]
        self.flow_blocks = nn.ModuleList([
            FlowBlock(
                in_channels * (patch_size ** 2),
                flow_block_dim,
                self.num_patches,
                self.permutations[i % 2],
                num_attn_blocks,
                attn_num_heads,
                attn_head_dim,
                ffn_expansion,
                num_classes,
            ) for i in range(num_flow_blocks)
        ])
    
    
    def patchify(self, x):
        B, C, H, W = x.shape
        S = self.patch_size
        h, w = H//S, W//S
        x = x.reshape(B, C, h, S, w, S)
        x = x.permute(0, 2, 4, 1, 3, 5) # (B, h, w, C, S, S)
        x = x.reshape(B, h*w, C*S*S)    # (B, T, c)
        return x
    
    
    def unpatchify(self, x):
        B, _, c = x.shape
        S = self.patch_size
        h = w = self.img_size // S
        C = c // (S**2)
        x = x.reshape(B, h, w, C, S, S)
        x = x.permute(0, 3, 1, 4, 2, 5) # (B, C, h, S, w, S)
        x = x.reshape(B, C, h*S, w*S)   # (B, C, H, W)
        return x        
        
        
    def forward(self, x, y):
        x = self.patchify(x)
        res_images = []
        accm_logdet = torch.zeros((), device=x.device)
        for flow_block in self.flow_blocks:
            x, log_det = flow_block(x, y)
            res_images.append(x)
            accm_logdet = accm_logdet + log_det
        return x, res_images, accm_logdet