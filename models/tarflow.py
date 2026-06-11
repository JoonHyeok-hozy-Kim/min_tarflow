
import torch
import torch.nn as nn

from models.attention import Transformer

class FlowBlock(nn.Module):
    def __init__(
        self, in_channels, flow_block_dim, num_patches, permutation, 
        num_attn_blocks, attn_num_heads, attn_head_dim, ffn_expansion,
        num_classes,
    ):
        super().__init__()
        self.hidden_dim = flow_block_dim
        self.output_dim = in_channels * 2   # NVP
        
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
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        
    def forward(self, x, y, attn_temperature=1.0):
        x = self.permutation(x)
        pos_embed_perm = self.permutation(self.pos_embed, dim=0)
        x_perm = x
        
        x = self.wo_in(x) + pos_embed_perm
        if self.class_embed is not None:
            if y is not None:
                # Negative label for CFG
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)  # (Y, 1, 1)
                    class_embed = (1-m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
            else:
                class_embed = self.class_embed[y]
        
            x = x + class_embed
        
        for transformer in self.transformer_blocks:
            x = transformer(x, self.attn_mask, attn_temperature, cache_key=None)
        x = self.w_out(x)
        
        # Right shift for the <BOS>
        x = torch.cat([
            torch.zeros_like(x[:, 0]),  # (B, 1, ...) all in 0
            x[:, -1],                   # (B, T-1, ...) exclude the last one
        ], dim=1)                       # (B, T, ...)   [0, x_0, x_1, ..., x_(T-2)]
        
        mu, alpha = x.chunk(2, dim=-1)
    

class TarFlow(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        pass