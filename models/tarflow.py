
import torch
import torch.nn as nn

from models.attention import Transformer
from models.permutation import PermutationIdentity, PermutationFlip
from models.attention import Attention

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
        
    def forward(self, x, y, attn_temp=1.0):
        x = self.permutation(x)
        pos_embed_perm = self.permutation(self.pos_embed, dim=0)
        x_perm = x  # "=i"
        
        x = self.w_in(x) + pos_embed_perm
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
            x = transformer(x, self.attn_mask, attn_temp, cache_key=None)
        x = self.w_out(x)
        
        # Implement "<i" for alpha and mu by shifting one-slot right.
        x = torch.cat([
            torch.zeros_like(x[:, :1]),     # (B, 1, c) all in 0
            x[:, :-1],                      # (B, T-1, c) exclude the last one
        ], dim=1)                           # (B, T, c)   [0, x_0, x_1, ..., x_(T-2)]
        
        alpha, mu = x.chunk(2, dim=-1)      # "<i"
        scale = torch.exp(-alpha.float()).type(alpha.dtype)
        x = self.permutation((x_perm - mu) * scale, inverse=True)
        return x, -alpha.mean(dim=[1, 2])   # Return mean instead of sum (Normalize by T)
    
    
    def reverse(self, x, y, cfg_weight=0.0, attn_temp=1.0, annealed_guidance=False):
        x_perm = self.permutation(x)
        pos_embed_perm = self.permutation(self.pos_embed, dim=0)
        self.set_attn_sample_mode(set_sample=True)
        
        B, T, c = x.shape
        for i in range(T-1):
            z_alhpa, z_mu = self._reverse(x_perm, pos_embed_perm, i, y, attn_temp, cache_key="cond")    # "<i" (B, 1, c)
            if cfg_weight > 0.0:
                uncond_z_alpha, uncond_z_mu = self._reverse(x_perm, pos_embed_perm, i, None, attn_temp, cache_key="uncond")
                if annealed_guidance:
                    g = (i+1) / (T-1) * cfg_weight  # Tarflow Appendix B
                else:
                    g = cfg_weight
                z_alhpa = z_alhpa + g * (z_alhpa - uncond_z_alpha)
                z_mu = z_mu + g * (z_mu - uncond_z_mu)
            
            z_alhpa, z_mu = z_alhpa[:, 0], z_mu[:, 0]   # "<i" (B, c)            
            scale = torch.exp(z_alhpa.float()).type(z_alhpa.dtype)    # (B, c) : no T-dim
            x_perm[:, i+1] = x_perm[:, i+1] * scale + z_mu            # Update single T-row of x_perm
            
        self.set_attn_sample_mode(set_sample=False)
        x_reperm = self.permutation(x_perm, inverse=True)          # (B, T, c)
        
        return x_reperm
    
    
    def set_attn_sample_mode(self, set_sample=True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = set_sample
                m.reset_cache()
    
    
    def _reverse(self, x, pos_embed, i, y, attn_temp, cache_key):
        x_in = x[:, i:i+1]                          # "=i" (B, 1, c)
        z = self.w_in(x_in) + pos_embed[:, i:i+1]   #      (B, 1, c)
        if self.class_embed is not None:
            if y is not None:
                z = z + self.class_embed[y]
            else:
                z = z + self.class_embed.mean(dim=0)
        
        for transformer in self.transformer_blocks:
            # kv-cache instead of attn_mask. Accumulated kv-cache implements "<i"
            z = transformer(z, attn_mask=None, attn_temp=attn_temp, cache_key=cache_key) 
        z = self.w_out(z)                   # "<i" (B, 2, c)
        z_alhpa, z_mu = z.chunk(2, dim=-1)  # "<i" (B, 1, c)
        
        return z_alhpa, z_mu    
            
    

class TarFlowRaw(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,    # Needs assignment for the 2D case (Not for images case)
        num_flow_blocks,        
        flow_block_dim,
        permutation_type,
        num_attn_blocks, 
        attn_num_heads, 
        attn_head_dim, 
        ffn_expansion,
        num_classes,
    ):
        super().__init__()
        self.num_patches = num_patches  
        
        if permutation_type == "flip":
            perm_list = [PermutationIdentity() if i%2==0 else PermutationFlip() for i in range(num_flow_blocks)]
        elif permutation_type == "shuffle":
            NotImplementedError()
        else:
            NotImplementedError()
            
        self.flow_blocks = nn.ModuleList([
            FlowBlock(
                in_channels,
                flow_block_dim,
                self.num_patches,
                perm_list[i],
                num_attn_blocks,
                attn_num_heads,
                attn_head_dim,
                ffn_expansion,
                num_classes,
            ) for i in range(num_flow_blocks)
        ])
        
        
    def forward(self, x, y):
        res = []
        accm_logdet = torch.zeros((), device=x.device)
        for flow_block in self.flow_blocks:
            x, log_det = flow_block(x, y)
            res.append(x)
            accm_logdet = accm_logdet + log_det
        return x, res, accm_logdet
    
    
    def get_loss(self, z, log_dets):
        return 0.5 * z.pow(2).mean() - log_dets.mean()  # Return mean instead of sum (Normalize by T) 
    
    
    def reverse(self, x, y, cfg_weight=0.0, attn_temp=1.0, annealed_guidance=False):
        # x = x * self.var.sqrt()   # Needed only for the VP case?
        for flow_block in self.flow_blocks:
            x = flow_block.reverse(x, y, cfg_weight, attn_temp, annealed_guidance)
        return x


class TarFlowImage(TarFlowRaw):
    def __init__(
        self,
        in_channels,
        img_size,
        patch_size,
        num_flow_blocks,        
        flow_block_dim,
        permutation_type,
        num_attn_blocks, 
        attn_num_heads, 
        attn_head_dim, 
        ffn_expansion,
        num_classes,
    ):
        assert img_size % patch_size == 0, f"img_size(={img_size}) is not divided by patch_size(={patch_size})."
        num_patches = (img_size // patch_size) ** 2
        super().__init__(
            in_channels,
            num_patches,
            num_flow_blocks,        
            flow_block_dim,
            permutation_type,
            num_attn_blocks, 
            attn_num_heads, 
            attn_head_dim, 
            ffn_expansion,
            num_classes,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
    
    
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
        x = self.patchify(x)    # (B, T, c)
        x, res_raw, accm_logdet = super().forward(x, y)
        x = self.unpatchify(x)
        res_img = [self.unpatchify(res) for res in res_raw]
        
        return x, res_img, accm_logdet