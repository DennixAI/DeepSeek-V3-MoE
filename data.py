import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

def initialize_tokenizer(hf_token=None):
    try: tokenizer = AutoTokenizer.from_pretrained("gpt2", token=hf_token)
    except: tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class TinyStoriesStreamDataset(IterableDataset):
    def __init__(self, split, tokenizer, seq_len, dataset_name, hf_token=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset = load_dataset(dataset_name, split="train" if split == "train" else "validation", streaming=True, token=hf_token)

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        while True:
            try:
                text = next(iterator)['text']
                tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=1e6, truncation=False)
                tokens.append(self.tokenizer.eos_token_id)
                buffer.extend(tokens)
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len + 1:]
                    yield {'input_ids': torch.tensor(chunk[:-1]), 'labels': torch.tensor(chunk[1:])}
            except StopIteration:
                iterator = iter(self.dataset)