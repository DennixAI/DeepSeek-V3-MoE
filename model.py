import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

def apply_rotary_emb(xq, xk, freq_cis):
    xq_out = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_out = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = freq_cis[:xq.shape[1]]
    xq_out = torch.view_as_real(xq_out * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_out * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.kv_lora_rank = args.kv_lora_rank
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.w_kv_down = nn.Linear(args.dim, args.kv_lora_rank, bias=False)
        self.w_kv_up = nn.Linear(args.kv_lora_rank, 2 * (args.n_heads * self.head_dim), bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.register_buffer("freqs_cis", self.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len))

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, x, mask=None):
        B, T, C = x.shape
        xq = self.wq(x)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        
        latent_kv = self.w_kv_down(x)
        kv = self.w_kv_up(latent_kv)
        kv = kv.view(B, T, 2, self.n_heads, self.head_dim)
        xk, xv = kv.unbind(2)
        
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)
        
        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class DeepSeekMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.num_shared = args.num_shared_experts
        
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.dim, args.expert_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(args.expert_hidden_dim, args.dim, bias=False)
            ) for _ in range(self.num_shared)
        ])
        
        self.routed_experts = nn.ModuleList([
             nn.Sequential(
                nn.Linear(args.dim, args.expert_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(args.expert_hidden_dim, args.dim, bias=False)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])
        
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        density_1 = probs.mean(dim=0)
        density_1_proxy = logits.mean(dim=0)
        aux_loss = (density_1 * density_1_proxy).sum() * self.num_experts
        
        final_out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            mask = (top_k_indices == i)
            batch_mask = mask.any(dim=-1)
            
            if batch_mask.any():
                expert_input = x_flat[batch_mask]
                expert_out = self.routed_experts[i](expert_input)
                
                weight_mask = mask[batch_mask]
                weights = top_k_weights[batch_mask]
                
                expanded_out = torch.zeros(expert_input.shape[0], self.top_k, expert_input.shape[1], device=x.device)
                
                for k in range(self.top_k):
                    k_mask = weight_mask[:, k]
                    if k_mask.any():
                        k_weights = weights[:, k][k_mask].unsqueeze(-1)
                        final_out[batch_mask] += expert_out[k_mask] * k_weights

        total_out = shared_out + final_out
        return total_out.view(*original_shape), aux_loss

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.attn = MLA(args)
        self.ffn_norm = RMSNorm(args.dim)
        self.moe = DeepSeekMoE(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        h = x + self.dropout(self.attn(self.attn_norm(x)))
        moe_out, aux_loss = self.moe(self.ffn_norm(h))
        out = h + self.dropout(moe_out)
        return out, aux_loss

class DeepSeekV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.linear_layer = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.gradient_checkpointing = args.gradient_checkpointing
        
        if args.use_liger and HAS_LIGER:
            self.le_loss = LigerFusedLinearCrossEntropyLoss()
        
        self.embedding.weight = self.linear_layer.weight
        self.last_aux_loss = 0.0
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x, aux_loss = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = self.norm(x)
        self.last_aux_loss = total_aux_loss
        
        if self.args.use_liger and self.training:
            return x 
            
        return self.linear_layer(x)