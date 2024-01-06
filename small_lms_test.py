# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler
from tokenizers import Tokenizer

from tqdm.auto import tqdm

from model import BigramModel
# -----------------------------

# setup cuda
torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")
# -----------------------------

# load tokenizer
tokenizer_file = "data/tokenizer-TinyStories3.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# load model
checkpoint = torch.load("checkpoints/4head-1.456M-checkpoint-2.pt")
hyperparameters = checkpoint['hyperparameters']
model = BigramModel(hyperparameters, device).to(device)
model.load_state_dict(checkpoint['model'])
# -----------------------------

# generate text
context = torch.tensor([[314, 324, 66, 283, 14]], dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=512)[0].tolist()))
# -----------------------------