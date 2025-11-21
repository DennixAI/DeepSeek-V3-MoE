import os
from transformers import AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

class Tokenizer:
    def __init__(self) -> None:
        hf_token = os.getenv("HF_TOKEN")  # your HF token in .env
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            token=hf_token,
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def get(self):
        return self.tokenizer
