import torch
import torch.nn.functional as F
import math
import tqdm 
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

try:
    from muon import MuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    HAS_MUON = False

from config import ModelArgs, get_args
from model import DeepSeekV3
from data import prepare_dataset, initialize_tokenizer

load_dotenv() 

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return local_rank, world_size, rank, device

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_lr(it, model_args):
    if it < model_args.warmup_iters:
        return model_args.max_lr * (it + 1) / (model_args.warmup_iters + 1)
    if it > model_args.lr_decay_iters:
        return model_args.min_lr
    decay_ratio = (it - model_args.warmup_iters) / (model_args.lr_decay_iters - model_args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return model_args.min_lr + coeff * (model_args.max_lr - model_args.min_lr)

def train():
    args = get_args()
    model_args = ModelArgs()
    
    # --- HF TOKEN HANDLING ---
    # Prioritize env var, then config
    hf_token = os.getenv("HF_TOKEN") or model_args.hf_token
    if hf_token:
        model_args.hf_token = hf_token
        print(f"✅ Loaded HF_TOKEN from environment.")
    else:
        print(f"⚠️ WARNING: No HF_TOKEN found. Uploads will fail.")

    # Initialize HF API and Create Repo if needed
    hf_api = HfApi(token=hf_token)
    if model_args.hf_repo_id and hf_token:
        try:
            create_repo(model_args.hf_repo_id, repo_type="model", exist_ok=True, token=hf_token)
            print(f"🔗 Connected to Hugging Face Repo: {model_args.hf_repo_id}")
        except Exception as e:
            print(f"⚠️ Could not create/connect to HF repo: {e}")

    tokenizer = initialize_tokenizer(model_args.hf_token)
    
    use_ddp = 'RANK' in os.environ or model_args.use_ddp
    if use_ddp:
        local_rank, world_size, rank, device = setup_ddp()
    else:
        device_type = model_args.device
        if device_type == "cuda" and not torch.cuda.is_available():
            device_type = "cpu"
        device = torch.device(device_type)
        rank = 0
        world_size = 1
        local_rank = 0
        
    if rank == 0:
        print(f"Training on {device} | World Size: {world_size}")
        print(f"Dataset: {model_args.dataset} ({model_args.dataset_subset})")
        print(f"Vocab Size: {model_args.vocab_size}")
    
    model = DeepSeekV3(model_args).to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    base_model = model.module if use_ddp else model
    
    def amp_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    hidden_weights = []
    norm_bias_params = []
    non_hidden_params = []
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad: continue
        if "embedding" in name or "linear_layer" in name:
            non_hidden_params.append(param)
        elif param.ndim < 2 or "norm" in name or "bias" in name:
            norm_bias_params.append(param)
        else:
            hidden_weights.append(param)

    param_groups = [
        {'params': hidden_weights, 'use_muon': True, 'lr': 0.02, 'weight_decay': 0.01},
        {'params': non_hidden_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': model_args.weight_decay_optim},
        {'params': norm_bias_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': 0.0}
    ]
    
    if HAS_MUON:
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        cleaned_groups = []
        for group in param_groups:
            cleaned_group = {k: v for k, v in group.items() if k != "use_muon"}
            cleaned_groups.append(cleaned_group)
        optimizer = torch.optim.AdamW(cleaned_groups)
        
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    # Dataset Loader (Updated to use subset name)
    train_dataloader = prepare_dataset('train', device, model_args.batch_size, use_ddp=use_ddp, tokenizer=tokenizer, model_args=model_args)
    train_iterator = iter(train_dataloader)
    
    val_dataloader = prepare_dataset('val', device, model_args.batch_size, use_ddp=use_ddp, tokenizer=tokenizer, model_args=model_args)
    val_iterator = iter(val_dataloader)
    
    train_history = []
    val_history = []

    def compute_loss(output, targets, model_ref):
        if model_args.use_liger and hasattr(model_ref, 'le_loss'):
            decoder_out_flat = output.contiguous().view(-1, model_args.dim)
            targets_flat = targets.contiguous().view(-1)
            return model_ref.le_loss(model_ref.linear_layer.weight, decoder_out_flat, targets_flat)
        else:
            logits_flat = output.contiguous().view(-1, model_args.vocab_size)
            targets_flat = targets.contiguous().view(-1)
            return F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(model_args.eval_iters):
            try:
                batch = next(val_iterator)
            except StopIteration:
                break
            idx, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            with amp_context():
                out = base_model(idx)
                l = compute_loss(out, targets, base_model)
            losses.append(l.item())

        if not losses: return 0.0
        avg_loss = torch.tensor(losses).mean().to(device)
        if use_ddp: dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        model.train()
        return avg_loss.item()

    model.train()
    if rank == 0:
        pbar = tqdm.tqdm(range(model_args.total_iters))
    
    accumulated_loss = 0.0
    
    for step in range(model_args.total_iters):
        lr = get_lr(step, model_args)
        # Update LR for non-Muon groups (index 1 and 2)
        for i in range(1, len(optimizer.param_groups)): 
            optimizer.param_groups[i]['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        
        for micro_step in range(model_args.gradient_accumulation_steps):
            do_sync = (micro_step == model_args.gradient_accumulation_steps - 1)
            if use_ddp and not do_sync:
                context = model.no_sync()
            else:
                context = nullcontext()

            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
                
            idx, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            
            with context:
                with amp_context():
                    output = model(idx)
                    ref_model = model.module if use_ddp else model
                    
                    cls_loss = compute_loss(output, targets, ref_model)
                    aux_loss = ref_model.last_aux_loss
                    
                    loss = cls_loss + (model_args.aux_loss_coef * aux_loss)
                    loss = loss / model_args.gradient_accumulation_steps
                    
                loss.backward()
            
            accumulated_loss += loss.item()
        
        if use_ddp:
            loss_tensor = torch.tensor(accumulated_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            accumulated_loss = loss_tensor.item()
            
        if model_args.clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.clip)
            
        optimizer.step()
        
        if rank == 0:
            pbar.update(1)
            pbar.set_description(f"Loss: {accumulated_loss:.4f}")
            train_history.append((step, accumulated_loss))
            
            # --- CHECKPOINT SAVING & UPLOADING ---
            if step % model_args.save_checkpoint_iter == 0 and step > 0:
                ckpt_name = f"checkpoint_{step}.pt"
                print(f"Saving checkpoint: {ckpt_name}")
                
                # Save locally
                torch.save({
                    'model': base_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'args': model_args
                }, ckpt_name)
                
                # Upload to HF
                if hf_token and model_args.hf_repo_id:
                    print(f"☁️ Uploading {ckpt_name} to Hugging Face...")
                    try:
                        hf_api.upload_file(
                            path_or_fileobj=ckpt_name,
                            path_in_repo=f"checkpoints/{ckpt_name}",
                            repo_id=model_args.hf_repo_id,
                            repo_type="model"
                        )
                        print("✅ Upload complete.")
                    except Exception as e:
                        print(f"❌ Upload failed: {e}")

            if step % model_args.eval_iters == 0 and step > 0:
                val_loss = estimate_loss()
                print(f"Validation Loss: {val_loss:.4f}")
                val_history.append((step, val_loss))

    if use_ddp:
        cleanup_ddp()
        
    if rank == 0:
        print("Training complete. Saving final model...")
        final_ckpt = "final_model.pt"
        torch.save(base_model.state_dict(), final_ckpt)
        
        if hf_token and model_args.hf_repo_id:
             try:
                hf_api.upload_file(
                    path_or_fileobj=final_ckpt,
                    path_in_repo="final_model.pt",
                    repo_id=model_args.hf_repo_id,
                    repo_type="model"
                )
                print("✅ Final model uploaded to Hugging Face.")
             except Exception as e:
                print(f"❌ Final upload failed: {e}")

if __name__ == "__main__":
    train()