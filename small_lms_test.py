# imports
import torch
from tokenizers import Tokenizer
from nano_gpt_model import NanoGPT
# -----------------------------

# setup cuda
torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")
# -----------------------------

# load tokenizer
tokenizer_file = "data/TinyStories-tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# load model
checkpoint = torch.load("checkpoints/simple-stories-4M.pt")
hyperparameters = checkpoint['hyperparameters']
model = NanoGPT(hyperparameters, device).to(device)
model.load_state_dict(checkpoint['model'])
# -----------------------------

# generate text
context = torch.tensor([[314, 324, 66, 283, 14]], dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=256)[0].tolist()))
# -----------------------------