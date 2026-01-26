from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    # --- Architecture Dimensions ---
    dim: int = 768                 
    n_layers: int = 16              
    n_heads: int = 12               # 768 / 64 = 12 heads
    n_kv_heads: int = 4             
    # Standard GQA (Grouped Query Attention)
    
    # --- MoE (Mixture of Experts) Config ---
    num_experts: int = 8            # Reduced from 64. Standard for small MoEs.
    num_shared_experts: int = 2     # Always active experts
    top_k: int = 2                  # Activate 2 experts per token (Standard)
    expert_hidden_dim: int = 2048   # 768 * 2.66 (Standard expansion for MoE is 2x-4x)
    
    # --- MLA (Memory Efficient Attention) ---
    kv_lora_rank: int = 128         # Compression for Key/Value heads
    q_lora_rank: int = 512          # Compression for Query heads
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    
    # --- Training & Data ---
    vocab_size: int = 50257         # for tinystories
    max_seq_len: int = 512
    batch_size: int = 16            # Effective batch size per GPU
    
    # --- Optimization ---
    lr_decay_iters: int = 50000     # Adjusted for shorter run
    warmup_iters: int = 1000
    max_lr: float = 4e-4            # Slightly higher LR for smaller models
    min_lr: float = 4e-5
    weight_decay_optim: float = 0.1
    clip: float = 1.0
    gradient_accumulation_steps: int = 4 # 8 * 4 = 32 effective batch size
    total_iters: int = 50000
    eval_iters: int = 200
    save_checkpoint_iter: int = 1000
    dropout: float = 0.0
    
    # --- System ---
    device: str = "cuda"
    use_ddp: bool = False           # Set to False if running on 1 GPU
    use_liger: bool = True          
    dataset: str = "HuggingFaceFW/fineweb-edu" 
    hf_token: str = None
    gradient_checkpointing: bool = True # Must be True to fit 14GB
    aux_loss_coef: float = 0.01

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    return args