# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


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
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
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


class Attention(nn.Module):
    def __init__(self, dim, n_heads, use_layer_norm, use_rope_2d):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads
        self.use_layer_norm = use_layer_norm
        self.use_rope_2d = use_rope_2d

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # LayerNorm
        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        
        # HookPoints for Q, K, V
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        
        # HookPoints for LayerNorm
        self.hook_q_norm = HookPoint()  # LayerNorm applied to Q
        self.hook_k_norm = HookPoint()  # LayerNorm applied to K
        
        # HookPoints by Heads
        self.hook_q_by_head = HookPoint()
        self.hook_k_by_head = HookPoint()
        self.hook_v_by_head = HookPoint()
        
        # HookPoints for Rope
        self.hook_rope_q = HookPoint()    # Rotary embedding applied to Q
        self.hook_rope_k = HookPoint()    # Rotary embedding applied to K
        
        self.hook_A_raw = HookPoint()
        self.hook_A_softmax = HookPoint()
        self.hook_Av = HookPoint()
        self.hook_output = HookPoint()

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out    

    @staticmethod
    def apply_rotary_emb_individually(x, freqs_cis, dtype):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis_x = Attention.reshape_for_broadcast(freqs_cis, x_)
        x_out = torch.view_as_real(x_ * freqs_cis_x).flatten(3)
        x_out = x_out.to(dtype)
        return x_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        
        # Q, K, V projections with hooks
        xq = self.hook_q(self.wq(x))
        xk = self.hook_k(self.wk(x))
        xv = self.hook_v(self.wv(x))

        dtype = xq.dtype

        # LayerNorm Hook
        if self.use_layer_norm:
            xq = self.hook_q_norm(self.q_norm(xq))
            xk = self.hook_k_norm(self.k_norm(xk))
        else:
            xq = self.hook_q_norm(xq)
            xk = self.hook_k_norm(xk)

        xq = self.hook_q_by_head(xq.view(bsz, seqlen, self.n_heads, self.head_dim))
        xk = self.hook_k_by_head(xk.view(bsz, seqlen, self.n_heads, self.head_dim))
        xv = self.hook_v_by_head(xv.view(bsz, seqlen, self.n_heads, self.head_dim))

        # Rope for Q and K with hooks
        if self.use_rope_2d:
            xq = self.hook_rope_q(self.apply_rotary_emb_individually(xq, freqs_cis, dtype))
            xk = self.hook_rope_k(self.apply_rotary_emb_individually(xk, freqs_cis, dtype))
        else:
            xq = self.hook_rope_q(xq)
            xk = self.hook_rope_k(xk)
        
        xq_permuted = xq.permute(0, 2, 1, 3)    # (B,H,S,D)
        xk_permuted = xk.permute(0, 2, 1, 3)
        xv_permuted = xv.permute(0, 2, 1, 3)        
        
        # Scaled dot-product attention from pytorch
        L, S = xq_permuted.size(-2), xk_permuted.size(-2)
        scale_factor = 1 / math.sqrt(xq_permuted.size(-1))
        
        # [Potential Attention Mask] : Multi-patch -inf로 보내주기!        
        # Raw attention weights with hook   
        attn_weight = self.hook_A_raw(
            xq_permuted @ xk_permuted.transpose(-2, -1) * scale_factor  # (B, H, S, S)
        )
        
        # No bias for now
        attn_bias = torch.zeros(L, S, dtype=xq_permuted.dtype, device=xq_permuted.device)        
        attn_weight = attn_weight + attn_bias   
        
        # Softmaxed attention weights with hook
        attn_weight = self.hook_A_softmax(torch.softmax(attn_weight, dim=-1))
        
        # Attention output with hook
        Av = self.hook_Av(attn_weight @ xv_permuted)    # (B, H, S, D)
        Av = Av.permute(0, 2, 1, 3)
        Av = Av.flatten(-2)     # (B, S, H*D)

        # Final output projection with hook
        output = self.hook_output(self.wo(Av))
        
        return output


class GeLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
    # def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        
        self.w_in = nn.Linear(dim, hidden_dim, bias=False)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.b_out = nn.Parameter(torch.zeros(dim))

        self.hook_pre_activation = HookPoint()
        self.hook_post_activation = HookPoint() # Also used for the neutralization
        
    def forward(self, x):
        x = self.hook_pre_activation(self.w_in(x) + self.b_in)
        x = F.gelu(x)
        x = self.hook_post_activation(x)
        x = self.w_out(x) + self.b_out
        return x
    

class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        self.hook_silu_gating = HookPoint()
        self.hook_out = HookPoint()

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        x = self.hook_silu_gating(self._forward_silu_gating(self.w1(x), self.w3(x)))
        x = self.hook_out(self.w2(x))
        return x
    

class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        d_ff=None,
        use_layer_norm=False,
        use_rope_2d=False,
        use_ffn_residual=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, use_layer_norm, use_rope_2d)
        # Use d_ff if provided, otherwise use 4 * dim
        hidden_dim = d_ff if d_ff is not None else 4 * dim
        self.feed_forward = GeLUFeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
        )
        self.layer_id = layer_id
        self.use_layer_norm = use_layer_norm            
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.use_ffn_residual = use_ffn_residual

        # adaLN_modulation output size: 6*dim for full block, 3*dim for only attention
        adaln_output_dim = 6 * dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), adaln_output_dim, bias=True),
        )
        
        # HookPoints for each block
        self.hook_attn_norm = HookPoint()
        self.hook_attn_adaLN = HookPoint()  # Right after Embedding
        self.hook_attn = HookPoint()
        self.hook_attn_gating = HookPoint()
        
        self.hook_ffn_norm = HookPoint()
        self.hook_ffn_adaLN = HookPoint()
        self.hook_ffn = HookPoint()
        self.hook_ffn_gating = HookPoint()

    def forward(self, x, freqs_cis, adaln_input):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(adaln_input).chunk(6, dim=1)
        )
        
        # x = x + gate_msa.unsqueeze(1) * self.attention(
        #     modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
        # )
        
        # Hooked attention block
        if self.use_layer_norm:
            x = self.hook_attn_norm(self.attention_norm(x))
        else:
            x = self.hook_attn_norm(x)
        attn_adaLN = self.hook_attn_adaLN(modulate(x, shift_msa, scale_msa))    # Embedding : x + t (adaLN)
        attn = self.hook_attn(self.attention(attn_adaLN, freqs_cis))
        attn_gated = self.hook_attn_gating(x + gate_msa.unsqueeze(1) * attn)
        
        # x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
        #     modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        # )
        
        # Hooked feed-forward block
        if self.use_layer_norm:
            attn_gated = self.hook_ffn_norm(self.ffn_norm(attn_gated))
        else:
            attn_gated = self.hook_ffn_norm(attn_gated)
        ffn_adaLN = self.hook_ffn_adaLN(modulate(attn_gated, shift_mlp, scale_mlp))
        ffn = self.hook_ffn(self.feed_forward(ffn_adaLN))
        
        # # Residual Connection
        if self.use_ffn_residual:
            ffn_gated = self.hook_ffn_gating(attn_gated + gate_mlp.unsqueeze(1) * ffn)
        else:
            ffn_gated = self.hook_ffn_gating(gate_mlp.unsqueeze(1) * ffn)
        
        # ffn_gated = ffn_adaLN

        return ffn_gated


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_layer_norm):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        
        # HookPoints for final layer
        self.hook_norm = HookPoint()
        self.hook_adaLN_norm = HookPoint()
        self.hook_output = HookPoint()
        
        # Moments HookPoints for Reverse Engineering
        self.moments_hook_norm_final = HookPoint()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        x = self.moments_hook_norm_final(x) # Moments recorded for the reverse LN
        
        # x = modulate(self.norm_final(x), shift, scale)
        # x = self.linear(x)
        
        # Hooked final layer (MLP)
        if self.use_layer_norm:
            x = self.hook_norm(self.norm_final(x))
        else:
            x = self.hook_norm(x)
            
        adaLN_normed = self.hook_adaLN_norm(modulate(x, shift, scale))
        output = self.hook_output(self.linear(adaLN_normed))        
        return output


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        only_attention=False,
        d_ff=None,
        dropout=0.0,
        use_layer_norm=False,
        use_rope_2d=False,
        use_cache=False,
        use_ffn_residual=False,
        prediction_target="x",       # Newly added for the v-prediction test
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        
        assert prediction_target in ("x", "v"), f"Unexpected prediction_target={prediction_target}"
        self.prediction_target = prediction_target

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    # only_attention=only_attention,
                    d_ff=d_ff,
                    # dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_rope_2d=use_rope_2d,
                    use_ffn_residual=use_ffn_residual,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, use_layer_norm=use_layer_norm)
        self.n_heads = n_heads
        self.dim = dim
        self.freqs_cis = None  # Will be computed dynamically based on input size
        
        # Give name to all HookPoints in its descendant modules
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = self.input_size // p
        w = (self.input_size * 3) // p  # 3x width after concatenation
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)   # (B, H//p * W//p, C*p*p)
        return x

    def forward(self, x, t, y):
        input_x = x.clone()

        src_channels = x.size(1)
        src_height = x.size(2)
        src_width = x.size(3)

        # Concatenate along W dimension (dim=3)
        x = torch.cat([x, y], dim=3)  # Now shape is (B, C, H, 3*W)
        
        x = self.patchify(x)    # (B, H//p*W//p, C*p*p)
        
        # Compute 2D positional embeddings for the current spatial size
        B, num_patches, _ = x.shape
        H_patches = src_height // self.patch_size
        W_patches = (src_width * 3) // self.patch_size  # 3x width after concatenation
        
        if self.freqs_cis is None or self.freqs_cis.size(0) != num_patches:
            self.freqs_cis = DiT_Llama.precompute_freqs_cis_2d(
                self.dim // self.n_heads, 
                H_patches, 
                W_patches
            ).to(x.device)
        
        self.freqs_cis = self.freqs_cis.to(x.device)
        
        x = self.x_embedder(x)  # (B, H//p*W//p, dim)
        t = self.t_embedder(t)  # (N, D)
        adaln_input = t.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, 3*W)
        
        # Precition target newly added!
        if self.prediction_target == "x":
            return input_x - x[:, :src_channels, :src_height, :src_width]       
        elif self.prediction_target == "v":
            return x[:, :src_channels, :src_height, :src_width]
        else:
            raise NotImplementedError(f"Unexpected prediction_target={self.prediction_target}")

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis_2d(dim, height, width, theta=10000.0):
        """
        Precompute 2D rotary positional embeddings for height and width.
        dim: dimension to split between height and width
        height: number of patches in height dimension
        width: number of patches in width dimension (3x original width)
        """
        dim_h = dim // 2  # half for height
        dim_w = dim - dim_h  # half for width
        
        # Height frequencies
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        t_h = torch.arange(height)
        freqs_h = torch.outer(t_h, freqs_h).float()
        freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
        
        # Width frequencies
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        t_w = torch.arange(width)
        freqs_w = torch.outer(t_w, freqs_w).float()
        freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
        
        # Create 2D grid
        freqs_cis_h = freqs_cis_h.unsqueeze(1).repeat(1, width, 1)  # (H, W, dim_h/2)
        freqs_cis_w = freqs_cis_w.unsqueeze(0).repeat(height, 1, 1)  # (H, W, dim_w/2)
        
        # Concatenate and flatten to (H*W, dim/2)
        freqs_cis_2d = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1)
        freqs_cis_2d = freqs_cis_2d.reshape(-1, dim // 2)
        
        return freqs_cis_2d
    
    
    # Methods for HookPoint management
    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
        
    def hook_points(self):
        return [module for name, module in self.named_modules() if ('hook' in name and ('moments' not in name or 'inject' not in name))]
    
    def moments_hook_points(self):
        return [module for name, module in self.named_modules() if 'moments' in name]
    
    def neutralize_frequency_hook_points(self):
        return [module for name, module in self.named_modules() if 'neutralize' in name]
    
    def inject_hook_points(self):
        return [module for name, module in self.named_modules() if 'inject' in name]
    
    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            if 'moments' in name:
                cache[name] = tensor
            elif 'inject' in name:
                pass
            else:
                cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
                
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')
        
        for mhp in self.moments_hook_points():
            mhp.add_hook(save_hook, 'moments')
            
    def enable_frequency_ablation(self, vectors):
        """
        vectors: A list of vectors u_k, v_k that will be neutralized.
        """
        for name, module in self.named_modules():
            if "hook_post_activation" in name:
                module.neutralize_target_vectors = vectors
                module.add_hook(None, dir='neutralize')

    def disable_frequency_ablation(self):
        for name, module in self.named_modules():
            if "hook_post_activation" in name:
                module.neutralize_target_vectors = None
                module.remove_hooks('fwd')


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)



if __name__ == "__main__":
    model = DiT_Llama_600M_patch2()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        out = model.forward_with_cfg(x, t, y, 0.5)
        print(out.shape)
