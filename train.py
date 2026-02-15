import os, time, glob, re, tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, create_repo, hf_hub_download
from config import ModelArgs
from model import DeepSeekV3
from data import TinyStoriesStreamDataset, initialize_tokenizer

# Optimization Helpers
try:
    from muon import MuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    HAS_MUON = False

# Utility Functions
def log_metrics(path, step, loss, lr):
    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        if not file_exists: f.write("step,loss,lr\n")
        f.write(f"{step},{loss},{lr}\n")

def log_generated_story(path, step, story):
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists: f.write("# 📖 DeepSeek-V3 TinyStories Generation Log\n\n---\n\n")
        f.write(f"## Step {step}\n```\n{story}\n```\n_Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n---\n\n")

@torch.no_grad()
def generate_story(model, tokenizer, prompt, max_new_tokens=100):
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -model.args.max_seq_len:]
        logits = model(input_cond)[:, -1, :]
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

@torch.no_grad()
def estimate_loss(model, val_iterator, val_loader, args):
    model.eval()
    losses = []
    for _ in range(args.val_batches):
        try: batch = next(val_iterator)
        except StopIteration:
            val_iterator = iter(val_loader); batch = next(val_iterator)
        idx, targets = batch['input_ids'].to(args.device), batch['labels'].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(idx)
            losses.append(F.cross_entropy(out.view(-1, args.vocab_size), targets.view(-1)).item())
    model.train()
    return sum(losses)/len(losses), val_iterator

def train():
    torch.cuda.empty_cache()
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', '12355'
        dist.init_process_group(backend='nccl', rank=0, world_size=1)

    args = ModelArgs()
    hf_api = HfApi(token=args.hf_token)
    if args.hf_repo_id: create_repo(args.hf_repo_id, repo_type="model", exist_ok=True, token=args.hf_token)

    local_csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    local_stories_path = os.path.join(args.checkpoint_dir, "generated_samples.md")

    # Download existing logs for continuity
    if args.hf_repo_id:
        for f in ["training_log.csv", "generated_samples.md"]:
            try: hf_hub_download(repo_id=args.hf_repo_id, filename=f, local_dir=args.checkpoint_dir, token=args.hf_token)
            except: pass

    tokenizer = initialize_tokenizer(args.hf_token)
    model = DeepSeekV3(args).to(args.device)
    
    # Param Grouping for Muon
    hidden, other = [], []
    for n, p in model.named_parameters():
        if p.requires_grad: (hidden if p.ndim >= 2 and "norm" not in n and "embedding" not in n else other).append(p)

    optimizer = MuonWithAuxAdam([
        {'params': hidden, 'use_muon': True, 'lr': 0.02},
        {'params': other, 'use_muon': False, 'lr': args.max_lr}
    ]) if HAS_MUON else torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    # Resume
    files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
    start_step = 0
    if files:
        latest = max(files, key=lambda f: int(re.search(r'checkpoint_(\d+).pt', f).group(1)))
        start_step = int(re.search(r'checkpoint_(\d+).pt', latest).group(1))
        ckpt = torch.load(latest, map_location=args.device)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])

    # Data
    train_loader = DataLoader(TinyStoriesStreamDataset('train', tokenizer, args.max_seq_len, args.dataset), batch_size=args.batch_size)
    val_loader = DataLoader(TinyStoriesStreamDataset('validation', tokenizer, args.max_seq_len, args.dataset), batch_size=args.batch_size)
    train_iter, val_iter = iter(train_loader), iter(val_loader)

    pbar = tqdm.tqdm(range(start_step, args.total_iters))
    for step in range(start_step, args.total_iters):
        lr = args.max_lr # Simplification for modularity
        optimizer.zero_grad()
        for _ in range(args.gradient_accumulation_steps):
            batch = next(train_iter)
            idx, targets = batch['input_ids'].to(args.device), batch['labels'].to(args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(idx)
                loss = (F.cross_entropy(out.view(-1, args.vocab_size), targets.view(-1)) + args.aux_loss_coef * model.last_aux_loss) / args.gradient_accumulation_steps
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if step % 500 == 0:
            val_loss, val_iter = estimate_loss(model, val_iter, val_loader, args)
            story = generate_story(model, tokenizer, "Once upon a time")
            log_generated_story(local_stories_path, step, story)
            print(f"Step {step} | Val Loss: {val_loss:.4f}")

        if step % 10 == 0: log_metrics(local_csv_path, step, loss.item()*args.gradient_accumulation_steps, lr)
        
        if step % args.save_checkpoint_iter == 0 and step > start_step:
            path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}, path)
            if args.hf_repo_id:
                for f, p in [(f"checkpoint_{step}.pt", path), ("training_log.csv", local_csv_path), ("generated_samples.md", local_stories_path)]:
                    hf_api.upload_file(path_or_fileobj=p, path_in_repo=f if ".pt" in f else f, repo_id=args.hf_repo_id)

if __name__ == "__main__":
    train()