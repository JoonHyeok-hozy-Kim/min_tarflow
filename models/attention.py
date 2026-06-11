
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, 3 * num_heads * head_dim)
        self.w_out = nn.Linear(num_heads * head_dim, in_channels)
        
        self.sample = False
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, mask, temperature, cache_key="cond"):
        B, T, _ = x.shape
        x = self.layer_norm(x.float()).type(x.dtype)
        qkv = self.qkv(x).reshape(B, T, 3 * self.num_heads, self.head_dim).transpose(1, 2)  # (B, 3H, T, D)
        q, k, v = qkv.chunk(3, dim=1)   # (B, H, T, D)
        
        if self.sample and cache_key is not None:
            assert self.k_cache is not None, f"k_cache not initialized"
            assert cache_key in self.k_cache, f"Unexpected cache_key : {cache_key}"
            
            self.k_cache[cache_key].append(k)
            self.v_cache[cache_key].append(v)
            k = torch.cat(self.k_cache[cache_key], dim=2)   # (B, H, i*T, D)
            v = torch.cat(self.v_cache[cache_key], dim=2)   # (B, H, i*T, D)
        
        scale = self.head_dim ** -0.5 / temperature
        if mask is not None:
            mask = mask.bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)   # (B, T, H*D)
        x = self.w_out(x)
        return x
    
    def reset_cache(self):
        self.k_cache = {'cond': [], 'uncond': []}
        self.v_cache = {'cond': [], 'uncond': []}        


class FeedForward(nn.Module):
    def __init__(self, in_channels, expansion):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.w_in = nn.Linear(in_channels, in_channels * expansion)
        self.activation = nn.GELU()
        self.w_out = nn.Linear(in_channels * expansion, in_channels)
        
    def forward(self, x):
        x = self.layer_norm(x.float()).type(x.dtype)
        x = self.w_in(x)
        x = self.activation(x)
        return self.w_out(x)
    

class Transformer(nn.Module):
    def __init__(self, in_channels, attn_num_heads, attn_head_dim, ffn_expansion=4):
        super().__init__()
        self.attention = Attention(in_channels, attn_num_heads, attn_head_dim)
        self.ffn = FeedForward(in_channels, ffn_expansion)
        
    def forward(self, x, attn_mask, attn_temp, cache_key):
        # Residual Connections
        x = x + self.attention(x, attn_mask, attn_temp, cache_key)
        x = x + self.ffn(x)
        return x