from dataclasses import dataclass
import torch
import os

@dataclass
class ModelArgs:
    # --- Architecture (Scaled for ~82M Params) ---
    dim: int = 384              
    n_layers: int = 6
    n_heads: int = 6               
    n_kv_heads: int = 2
    
    # --- MoE Config ---
    num_experts: int = 8            
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 1024
    
    # --- MLA Config ---
    kv_lora_rank: int = 64         
    q_lora_rank: int = 256   
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    
    # --- Training ---
    vocab_size: int = 50257         # GPT-2 Vocab
    max_seq_len: int = 512          
    batch_size: int = 16            # Physical batch per step
    gradient_accumulation_steps: int = 4 # Effective batch = 64
    
    # --- Optimization ---
    lr_decay_iters: int = 50000     
    warmup_iters: int = 1000
    max_lr: float = 6e-4            
    min_lr: float = 6e-5
    weight_decay_optim: float = 0.1
    clip: float = 1.0
    total_iters: int = 50000
    save_checkpoint_iter: int = 1000
    dropout: float = 0.0
    aux_loss_coef: float = 0.01
    
    # --- System & Infrastructure ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_liger: bool = False 
    dataset: str = "roneneldan/TinyStories" 
    checkpoint_dir: str = "checkpoints_tinystories"
    hf_token: str = "YOUR_TOKEN_HERE"
    hf_repo_id: str = "FusionCorp/DeepSeek-V3-TinyStories"
    gradient_checkpointing: bool = False

    # --- Evaluation ---
    val_interval: int = 500         
    val_batches: int = 20