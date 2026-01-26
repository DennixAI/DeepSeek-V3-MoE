import torch
import torch.nn.functional as F
import math
import tqdm 
import os
import glob
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, hf_hub_download

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

def find_latest_checkpoint(repo_id=None, token=None):
    # 1. Check Local Files first
    local_files = glob.glob("checkpoint_*.pt")
    if local_files:
        # Extract step numbers: "checkpoint_1000.pt" -> 1000
        latest_file = max(local_files, key=lambda f: int(re.search(r'checkpoint_(\d+).pt', f).group(1)))
        step_num = int(re.search(r'checkpoint_(\d+).pt', latest_file).group(1))
        print(f"📂 Found local checkpoint: {latest_file} (Step {step_num})")
        return latest_file, step_num

    # 2. If no local files, check Hugging Face
    if repo_id and token:
        print(f"🔍 No local checkpoints. Checking Hugging Face repo: {repo_id}...")
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            ckpt_files = [f for f in files if "checkpoint_" in f and f.endswith(".pt")]
            
            if ckpt_files:
                # Filter paths like "checkpoints/checkpoint_1000.pt" -> 1000
                latest_remote = max(ckpt_files, key=lambda f: int(re.search(r'checkpoint_(\d+).pt', f).group(1)))
                step_num = int(re.search(r'checkpoint_(\d+).pt', latest_remote).group(1))
                
                print(f"☁️ Found remote checkpoint: {latest_remote} (Step {step_num})")
                print("⬇️ Downloading checkpoint...")
                
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=latest_remote,
                    token=token
                )
                return local_path, step_num
        except Exception as e:
            print(f"⚠️ Could not check/download from HF: {e}")
    
    return None, 0

def train():
    args = get_args()
    model_args = ModelArgs()
    
    # --- HF SETUP ---
    hf_token = os.getenv("HF_TOKEN") or model_args.hf_token
    if hf_token: model_args.hf_token = hf_token
    
    hf_api = HfApi(token=hf_token)
    if model_args.hf_repo_id and hf_token:
        try:
            create_repo(model_args.hf_repo_id, repo_type="model", exist_ok=True, token=hf_token)
        except Exception:
            pass

    tokenizer = initialize_tokenizer(model_args.hf_token)
    model_args.vocab_size = len(tokenizer) 

    
    # DDP Setup
    use_ddp = 'RANK' in os.environ or model_args.use_ddp
    if use_ddp:
        local_rank, world_size, rank, device = setup_ddp()
    else:
        device_type = model_args.device
        if device_type == "cuda" and not torch.cuda.is_available(): device_type = "cpu"
        device = torch.device(device_type)
        rank = 0; world_size = 1; local_rank = 0
        
    if rank == 0:
        print(f"🚀 Training on {device} | World Size: {world_size}")

    # Initialize Model
    model = DeepSeekV3(model_args).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    base_model = model.module if use_ddp else model

    # Optimizer Setup
    hidden_weights, norm_bias_params, non_hidden_params = [], [], []
    for name, param in base_model.named_parameters():
        if not param.requires_grad: continue
        if "embedding" in name or "linear_layer" in name: non_hidden_params.append(param)
        elif param.ndim < 2 or "norm" in name or "bias" in name: norm_bias_params.append(param)
        else: hidden_weights.append(param)

    param_groups = [
        {'params': hidden_weights, 'use_muon': True, 'lr': 0.02, 'weight_decay': 0.01},
        {'params': non_hidden_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': model_args.weight_decay_optim},
        {'params': norm_bias_params, 'use_muon': False, 'lr': model_args.max_lr, 'weight_decay': 0.0}
    ]
    
    if HAS_MUON: optimizer = MuonWithAuxAdam(param_groups)
    else:
        cleaned = [{k:v for k,v in g.items() if k!='use_muon'} for g in param_groups]
        optimizer = torch.optim.AdamW(cleaned)
        
    if hasattr(torch, "compile"): model = torch.compile(model)

    # resume logic
    start_step = 0
    if rank == 0:
        ckpt_path, found_step = find_latest_checkpoint(model_args.hf_repo_id, hf_token)
        if ckpt_path:
            print(f"⏩ Resuming training from step {found_step}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            base_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = found_step
            # Clean up memory
            del checkpoint
            torch.cuda.empty_cache()
    
    # Broadcast start_step to other GPUs if DDP
    if use_ddp:
        step_tensor = torch.tensor([start_step], device=device)
        dist.broadcast(step_tensor, src=0)
        start_step = step_tensor.item()

    # Data Loader
    train_dataloader = prepare_dataset('train', device, model_args.batch_size, use_ddp=use_ddp, tokenizer=tokenizer, model_args=model_args)
    train_iterator = iter(train_dataloader)
    
    val_dataloader = prepare_dataset('val', device, model_args.batch_size, use_ddp=use_ddp, tokenizer=tokenizer, model_args=model_args)
    val_iterator = iter(val_dataloader)

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
            try: batch = next(val_iterator)
            except StopIteration: break
            idx, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            with amp_context():
                out = base_model(idx)
                l = compute_loss(out, targets, base_model)
            losses.append(l.item())
        if not losses: return 0.0
        avg = torch.tensor(losses).mean().to(device)
        if use_ddp: dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        model.train()
        return avg.item()

    def amp_context():
        if device.type == "cuda": return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    model.train()
    if rank == 0:
        # Start progress bar from start_step
        pbar = tqdm.tqdm(range(start_step, model_args.total_iters), initial=start_step, total=model_args.total_iters)
    
    accumulated_loss = 0.0
    
    # Loop starts from start_step now
    for step in range(start_step, model_args.total_iters):
        lr = get_lr(step, model_args)
        for i in range(1, len(optimizer.param_groups)): optimizer.param_groups[i]['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        
        for micro_step in range(model_args.gradient_accumulation_steps):
            do_sync = (micro_step == model_args.gradient_accumulation_steps - 1)
            if use_ddp and not do_sync: context = model.no_sync()
            else: context = nullcontext()

            try: batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
                
            idx, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            
            with context:
                with amp_context():
                    output = model(idx)
                    ref_model = model.module if use_ddp else model
                    cls_loss = compute_loss(output, targets, ref_model)
                    loss = (cls_loss + (model_args.aux_loss_coef * ref_model.last_aux_loss)) / model_args.gradient_accumulation_steps
                loss.backward()
            accumulated_loss += loss.item()
        
        if use_ddp:
            loss_tensor = torch.tensor(accumulated_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            accumulated_loss = loss_tensor.item()
            
        if model_args.clip > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.clip)
        optimizer.step()
        
        if rank == 0:
            pbar.update(1)
            pbar.set_description(f"Loss: {accumulated_loss:.4f}")
            
            if step % model_args.save_checkpoint_iter == 0 and step > start_step:
                ckpt_name = f"checkpoint_{step}.pt"
                # Save locally
                torch.save({
                    'model': base_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'args': model_args
                }, ckpt_name)
                
                # Upload to HF
                if hf_token and model_args.hf_repo_id:
                    try:
                        hf_api.upload_file(
                            path_or_fileobj=ckpt_name,
                            path_in_repo=f"checkpoints/{ckpt_name}",
                            repo_id=model_args.hf_repo_id,
                            repo_type="model"
                        )
                    except Exception as e: print(f"❌ Upload failed: {e}")

            if step % model_args.eval_iters == 0 and step > start_step:
                print(f"Validation Loss: {estimate_loss():.4f}")

    if use_ddp: cleanup_ddp()
        
    if rank == 0:
        print("Training complete.")
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
             except Exception: pass

if __name__ == "__main__":
    train()