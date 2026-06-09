import torch
import torch.nn as nn
import torch.nn.functional as F

from models.permutations import PERMUTATION_TYPES
from models.transformer import Transformer, Attention


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FlowBlock(nn.Module):    
    def __init__(
        self, 
        in_channels,            # D 
        seq_len,                # N
        flow_block_dim,
        num_attn_layers,
        permutation_class,
        attn_head_dim,
        attn_temp,
        ffn_multiplier,
        num_classes,
        class_dropout_prob,
    ):
        super().__init__()
                
        self.in_channels = in_channels
        self.out_channels = in_channels * 2     # alpha, mu = x_out.chunk(2, dim=-1)
        self.seq_len = seq_len
        self.flow_block_dim = flow_block_dim
        self.num_attn_layers = num_attn_layers
        self.permutation = permutation_class(self.seq_len)
        
        self.attn_temp = attn_temp
        self.freqs_cis = None
        
        self.w_in = nn.Linear(in_channels, flow_block_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, flow_block_dim) * 1e-2)
        self.y_embed = LabelEmbedder(num_classes, flow_block_dim, class_dropout_prob)
        self.attn_blocks = torch.nn.ModuleList([
            Transformer(flow_block_dim, attn_head_dim, ffn_multiplier) for _ in range(num_attn_layers)
        ])
        self.w_out = nn.Linear(flow_block_dim, self.out_channels)
        
        self.register_buffer('attn_mask', torch.tril(torch.ones(seq_len, seq_len))) # Attention Mask
        
        
    def forward(self, x, y):
        # Permutation
        x_perm = self.permutation(x, dim=1)                             # (B, seq_len', in_channels)
        permuted_pos_embed = self.permutation(self.pos_embed, dim=0)    # (seq_len', flow_block_dim)
        
        zx = self.w_in(x_perm)                                          # (B, seq_len', flock_block_dim)
        zpos = permuted_pos_embed.to(x.dtype).unsqueeze(0)              # (1, seq_len', flow_block_dim)
        zy = self.y_embed(y, self.training).to(x.dtype).unsqueeze(1)    # (B, 1, flow_block_dim)
        # print(f"y:{zy.shape}, x:{zx.shape}, pos:{zpos.shape}")
        z = zx + zpos + zy                                              # (B, seq_len', flow_block_dim)

        # TAR
        for transformer in self.attn_blocks:
            z = transformer(z, self.attn_mask, self.attn_temp, self.freqs_cis)
        
        z = self.w_out(z)                                               # (B, seq_len', in_channels * 2)
        z = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)   # [0, x_0, x_1, ..., x_{seq_len-1}]
        
        # TARFlow actually used NVP
        alpha, mu = z.chunk(2, dim=-1)
        scale = torch.exp(-alpha).to(z.dtype)
        
        z = (x_perm - mu) * scale        
        z = self.permutation(z, dim=1, inverse=True)       
        log_det = torch.mean(alpha, dim=[1,2])      # Use mean instead of sum for normalization
        
        return z, log_det
    
    def set_sample_mode_iteratively(self, flag):
        for module in self.modules():
            if isinstance(module, Attention):
                module.set_sample_mode(flag)
    
    def reverse(self, x, y):
        # Permutation and embedded
        x_perm = self.permutation(x, dim=1)                            # (B, seq_len', in_channels)
        permuted_pos_embed = self.permutation(self.pos_embed, dim=0)   
        permuted_pos_embed = permuted_pos_embed.unsqueeze(0)           # (1, seq_len', flow_block_dim)
        y = self.y_embed(y, self.training).to(x.dtype).unsqueeze(1)    # (B, 1, in_channels)
        
        # Set attentions to sample mode and initialize kv cache
        self.set_sample_mode_iteratively(flag=True)
        
        B, seq_len, _ = x.shape
        for i in range(seq_len-1):
            alpha_i, mu_i = self.reverse_step(x, permuted_pos_embed, i, y, which_cache="cond")
            scale_i = torch.exp(alpha_i).to(alpha_i.dtype)      # (B, 1, flow_block_dim)
            scale_i, mu_i = scale_i[:, 0], mu_i[:, 0]           # (B, flow_block_dim)
            
            x_perm[:, i+1] = x_perm[:, i+1] * scale_i + mu_i
        
        self.set_sample_mode_iteratively(flag=False)
        x = self.permutation(x_perm, dim=1, inverse=True)
        return x        
    
    def reverse_step(self, x, pos_embed, i, y, which_cache):
        x_i = x[:, i:i+1]   # i-th patch of x : Since mu in forward was causal, x_i depends on "<i"
        z_i = self.w_in(x_i) + pos_embed + y
        
        # TAR
        for transformer in self.attn_blocks:
            z_i = transformer(z_i, attn_mask=None, attn_temp=self.attn_temp, freqs_cis=self.freqs_cis, which_cache=which_cache)
        
        z_i = self.w_out(z_i)
        alpha_i, mu_i = z_i.chunk(2, dim=-1)
        
        return alpha_i, mu_i
    

class TarFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,       # C
        img_size: int,          # W, H
        patch_size: int,        # S
        num_flow_blocks: int,   # T
        flow_block_dim: int,    
        num_attn_layers: int,
        perumtation_type: str,  # ("flip", "shuffle")
        attn_head_dim: int,     # H
        attn_temp: int,
        ffn_multiplier: int,
        num_classes: int = 0,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, f"Invalid patch_size. (img_size={img_size}, patch_size={patch_size})."
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2    # Also equivalent to the seq_len
        self.patch_dim = in_channels * (patch_size ** 2)
        
        assert perumtation_type in PERMUTATION_TYPES.keys(), f"Unexpected perumtation_type={perumtation_type}."
        flow_block_list = []
        for t in range(num_flow_blocks):
            if t == 0:
                permutation_class = PERMUTATION_TYPES["identity"]
            else:
                permutation_class = PERMUTATION_TYPES[perumtation_type]
            
            flow_block_list.append(
                FlowBlock(
                    self.patch_dim,         # D = C * S^2
                    self.num_patches,       # N = H*W // S^2
                    flow_block_dim,         # flow_block's model dim
                    num_attn_layers,        # number of Attention layers for each Flow Block to stack
                    permutation_class,      # permutation class : will be instatited in the FlowBlock
                    attn_head_dim,          # head dim of each Attention layer
                    attn_temp,              # attention temperature
                    ffn_multiplier,         # multiplier for FFN's parameter size
                    num_classes,            # [hozy] Is it necessary? What if LabelEmbbedin is added once at TarFlow?
                    class_dropout_prob,     # for LabelEmbedder
                )
            )            
        
        self.flow_blocks = nn.ModuleList(flow_block_list)
    
    def patchify(self, x):
        B, C, H, W = x.size()
        S = self.patch_size
        h, w = H//S, W//S
        
        x = x.reshape(B, C, h, S, w, S)
        x = torch.einsum("bchswS->bhwcsS", x)
        x = x.reshape(B, h*w, C*S*S)            # (B, seq_len, patch_dim)
        return x
    
    def unpatchify(self, x):
        B = x.shape[0]
        C = self.out_channels
        S = self.patch_size
        h, w = self.img_size // S, self.img_size // S
        x = x.reshape(B, h, w, C, S, S)
        x = torch.einsum('bhwcsS->bchswS', x)
        x = x.reshape(B, C, h*S, w*S)
        return x
        
    def forward(self, x, y):
        x = self.patchify(x)    # (B, h, w, C*S^2)
        outputs = []
        total_log_det = torch.zeros((), device=x.device)
        for flow_block in self.flow_blocks:
            x, curr_log_det = flow_block(x, y)
            total_log_det = total_log_det + curr_log_det
            outputs.append(x)
        
        return x, outputs, total_log_det
    
    def update_prior(self, z):
        raise NotImplementedError()
    
    def get_loss(self, z, total_log_det):
        return 0.5 * torch.mean(z**2) - torch.mean(total_log_det)   # Use mean instead of sum for normalization in FlowBlock.forward
    
    def reverse(self, z, y):
        outputs = [self.unpatchify(z)]          # Store in (B, C, H, W)
        for flow_block in self.flow_blocks:
            z = flow_block.reverse(z, y)
            outputs.append(self.unpatchify(z))
        
        return outputs[-1], outputs