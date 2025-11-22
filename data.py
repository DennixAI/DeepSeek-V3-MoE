import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.distributed as dist
from config import ModelArgs

def initialize_tokenizer(hf_token=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=hf_token)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class FineWebStreamDataset(IterableDataset):
    def __init__(self, split, tokenizer, seq_len, batch_size, world_size=1, rank=0, infinite=True, dataset_name="HuggingFaceFW/fineweb-edu", dataset_config="sample-10BT", hf_token=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.world_size = world_size
        self.rank = rank
        self.infinite = infinite
        
        try:
            self.dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split="train",
                streaming=True,
                token=hf_token
            )
        except Exception:
             self.dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split="train",
                streaming=True,
                token=hf_token
            )

        if self.world_size > 1:
            self.dataset = self.dataset.shard(num_shards=self.world_size, index=self.rank)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds_iterator = iter(self.dataset)
            for _ in range(worker_info.id):
                next(ds_iterator, None)
            step_size = worker_info.num_workers
        else:
            ds_iterator = iter(self.dataset)
            step_size = 1

        buffer_tokens = []
        
        while True:
            try:
                if step_size > 1:
                    for _ in range(step_size - 1):
                        next(ds_iterator, None)
                
                example = next(ds_iterator)
                text = example['text']
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                buffer_tokens.extend(tokens)
                
                while len(buffer_tokens) >= self.seq_len + 1:
                    chunk = buffer_tokens[:self.seq_len + 1]
                    buffer_tokens = buffer_tokens[self.seq_len + 1:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {'input_ids': x, 'labels': y}
                    
            except StopIteration:
                if self.infinite:
                    ds_iterator = iter(self.dataset)
                else:
                    break

def prepare_dataset(split, device, batch_size, use_ddp=False, tokenizer=None, model_args=None):
    if use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    model_args = model_args or ModelArgs()
    tokenizer = tokenizer or initialize_tokenizer(model_args.hf_token)

    ds = FineWebStreamDataset(
        split=split,
        tokenizer=tokenizer, 
        seq_len=model_args.max_seq_len,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        infinite=(split == 'train'),
        dataset_name=model_args.dataset,
        hf_token=model_args.hf_token
    )
    
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)