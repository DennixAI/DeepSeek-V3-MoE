import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

def apply_rotary_emb(xq, xk, freq_cis):
    xq_out = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_out = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = freq_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1)
    xq_out = torch.view_as_real(xq_out * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_out * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim, self.n_heads = args.dim, args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.w_kv_down = nn.Linear(args.dim, args.kv_lora_rank, bias=False)
        self.w_kv_up = nn.Linear(args.kv_lora_rank, 2 * (args.n_heads * self.head_dim), bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.wo.res_scale = True # Tag for scaled init
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.register_buffer("freqs_cis", self.precompute_freqs_cis(self.head_dim, args.max_seq_len))

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x):
        B, T, C = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        kv = self.w_kv_up(self.w_kv_down(x)).view(B, T, 2, self.n_heads, self.head_dim)
        xk, xv = kv.unbind(2)
        xq, xk = apply_rotary_emb(self.q_norm(xq), self.k_norm(xk), self.freqs_cis)
        out = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))

class DeepSeekMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_experts, self.top_k = args.num_experts, args.top_k
        self.num_shared = args.num_shared_experts
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.dim, args.expert_hidden_dim, bias=False), nn.SiLU(), nn.Linear(args.expert_hidden_dim, args.dim, bias=False)) for _ in range(self.num_shared)])
        self.routed_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.dim, args.expert_hidden_dim, bias=False), nn.SiLU(), nn.Linear(args.expert_hidden_dim, args.dim, bias=False)) for _ in range(self.num_experts)])
        for e in list(self.shared_experts) + list(self.routed_experts): e[2].res_scale = True

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        probs = F.softmax(self.gate(x_flat), dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        aux_loss = (probs.mean(dim=0) ** 2).sum() * self.num_experts
        final_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            for k in range(self.top_k):
                mask = (indices[:, k] == i)
                if mask.any():
                    final_out[mask] += self.routed_experts[i](x_flat[mask]) * weights[mask, k].unsqueeze(-1)
        return (shared_out + final_out).view(*orig_shape), aux_loss

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(args.dim), MLA(args)
        self.ffn_norm, self.moe = RMSNorm(args.dim), DeepSeekMoE(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        h = x + self.dropout(self.attn(self.attn_norm(x)))
        out, aux = self.moe(self.ffn_norm(h))
        return h + self.dropout(out), aux

class DeepSeekV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm, self.linear_layer = RMSNorm(args.dim), nn.Linear(args.dim, args.vocab_size, bias=False)
        self.embedding.weight = self.linear_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        std = 0.02
        if hasattr(m, 'res_scale'): std *= (2 * self.args.n_layers) ** -0.5
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=std)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx):
        x, total_aux = self.embedding(idx), torch.tensor(0.0, device=idx.device)
        for l in self.layers:
            x, aux = torch.utils.checkpoint.checkpoint(l, x, use_reentrant=True) if self.training and self.args.gradient_checkpointing else l(x)
            total_aux = total_aux + aux
        self.last_aux_loss = total_aux
        return self.linear_layer(self.norm(x))