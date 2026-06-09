# Code based on https://github.com/cloneofsimo/minRF/blob/main/dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, attn_head_dim):
        super().__init__()
        self.num_attn_head = in_channels // attn_head_dim   # H
        self.attn_head_dim = attn_head_dim                  # D
        
        self.sample = False
        self.k_cache = None
        self.v_cache = None        
        
        self.wq = nn.Linear(in_channels, self.num_attn_head * self.attn_head_dim, bias=False)
        self.wk = nn.Linear(in_channels, self.num_attn_head * self.attn_head_dim, bias=False)
        self.wv = nn.Linear(in_channels, self.num_attn_head * self.attn_head_dim, bias=False)
        self.wo = nn.Linear(self.num_attn_head * self.attn_head_dim, in_channels, bias=False)
        
        self.q_norm = nn.LayerNorm(self.num_attn_head * self.attn_head_dim)
        self.k_norm = nn.LayerNorm(self.num_attn_head * self.attn_head_dim)
    
    def forward(self, x, attn_mask, attn_temp, freqs_cis, which_cache):
        B, seq_len, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = x.dtype
        
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        xq = xq.view(B, seq_len, self.num_attn_head, self.attn_head_dim).transpose(1, 2)    # (B, H, seqlen, D)
        xk = xk.view(B, seq_len, self.num_attn_head, self.attn_head_dim).transpose(1, 2)    # (B, H, seqlen, D)
        xv = xv.view(B, seq_len, self.num_attn_head, self.attn_head_dim).transpose(1, 2)    # (B, H, seqlen, D)
        
        # # [To-Do] RoPE implementation
        # xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # xq, xk = xq.to(dtype), xk.to(dtype)
        
        if self.sample:
            self.k_cache[which_cache].append(xk)
            self.v_cache[which_cache].append(xv)
            xk = torch.cat(self.k_cache[which_cache], dim=2)  # appended to seqlen
            xv = torch.cat(self.v_cache[which_cache], dim=2)            
        
        # Temperature Scaling
        dk = self.attn_head_dim ** (-0.5)
        dk = dk / attn_temp
        
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
        
        output = F.scaled_dot_product_attention(
            xq,                     # (B, H, seqlen, D)
            xk,                     # (B, H, seqlen, D)
            xv,                     # (B, H, seqlen, D)
            dropout_p=0.0,
            attn_mask=attn_mask,    # Provide manual causality!
            scale=dk,
        ).transpose(1, 2).reshape(B, seq_len, self.num_attn_head * self.attn_head_dim)
        
        return self.wo(output)
    
    def set_sample_mode(self, flag=True):
        self.sample = flag
        if flag:
            del self.k_cache, self.v_cache
            self.initialize_kv_cache()
    
    def initialize_kv_cache(self):
        self.k_cache = {'cond': [], 'uncond': []}
        self.v_cache = {'cond': [], 'uncond': []}        
    

class FeedForward(nn.Module):
    def __init__(self, in_channels, multiple_of):
        super().__init__()
        self.w1 = nn.Linear(in_channels, in_channels * multiple_of, bias=False)
        self.w2 = nn.GELU()
        self.w3 = nn.Linear(in_channels * multiple_of, in_channels, bias=False)
        
    def forward(self, x):
        x = self.w1(x.float()).type(x.dtype)
        x = self.w2(x.float()).type(x.dtype)
        x = self.w3(x.float()).type(x.dtype)
        return x
    

class Transformer(nn.Module):
    def __init__(self, in_channels, attn_head_dim, ffn_multiplier):
        super().__init__()
        self.attention = Attention(in_channels, attn_head_dim)
        self.ffn = FeedForward(in_channels, ffn_multiplier)
    
    def forward(self, x, attn_mask, attn_temp, freqs_cis, which_cache='cond'):
        x = x + self.attention(x, attn_mask, attn_temp, freqs_cis, which_cache)
        x = x + self.ffn(x)
        return x