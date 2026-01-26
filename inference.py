import torch
import torch.nn.functional as F
from model import DeepSeekV3
from config import ModelArgs
from data import initialize_tokenizer

def load_model(checkpoint_path, device="cuda"):
    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. Extract args from checkpoint or use default
    if 'args' in checkpoint:
        args = checkpoint['args']
    else:
        # Fallback to default TinyStories settings
        args = ModelArgs()
        args.vocab_size = 50257
        args.max_seq_len = 512

    # 3. Initialize model and load weights
    model = DeepSeekV3(args)
    
    # Handle both full checkpoint (with optimizer) and state_dict only
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Clean up 'module.' prefix if saved with DDP
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, args

@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=100, 
    temperature=1.0, 
    top_k=50, 
    device="cuda"
):
    """
    Generates text based on a prompt.
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop input if it exceeds max_seq_len
        input_cond = input_ids if input_ids.size(1) <= model.args.max_seq_len else input_ids[:, -model.args.max_seq_len:]
        
        # Forward pass
        # Note: model returns hidden states if use_liger is true in training, 
        # but for inference, e use the linear_layer manually if needed.
        # Based on our model.py, if not training, it returns logits.
        logits = model(input_cond)
        
        # Focus on the last token's logits
        logits = logits[:, -1, :] / temperature
        
        # Optional: Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

if __name__ == "__main__":
    # Configuration
    CHECKPOINT = "final_model.pt"  # or "checkpoint_1000.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {CHECKPOINT}...")
    tokenizer = initialize_tokenizer()
    model, args = load_model(CHECKPOINT, device=DEVICE)
    
    # Test Prompts
    prompts = [
        "Once upon a time, there was a little bird named",
        "Lily was very sad because",
        "The brave knight took his sword and"
    ]
    
    print("\n--- Starting Generation ---\n")
    for p in prompts:
        output = generate(model, tokenizer, p, max_new_tokens=100, temperature=0.8, top_k=40, device=DEVICE)
        print(f"Prompt: {p}")
        print(f"Output: {output}")
        print("-" * 30)