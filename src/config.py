import torch 

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SEED = 42
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32
MODEL_DTYPE = torch.float16

INPUT_FILE = "data/gsm8k_clean.jsonl"