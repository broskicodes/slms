# imports
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
# -----------------------------

# load dataset
dataset = load_dataset("roneneldan/TinyStories")

# write data to csv
concatenate_datasets([dataset["validation"], dataset['train']]).to_csv("data/stories.csv")

# convert csv file to txt file
input_file_path = "data/stories.csv"
df = pd.read_csv(input_file_path)

output_file_path = "data/stories.txt"
with open(output_file_path, 'w') as file:
    for row in df['text']:
        file.write(str(row) + '\n')
# -----------------------------

# initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="<|unknown|>"))
trainer = BpeTrainer(special_tokens=["<|unknown|>", "<|im_start|>", "<|im_end|>"], vocab_size=2048, min_frequency=1)
tokenizer.pre_tokenizer = Whitespace()
# -----------------------------

# train and save tokenizer
tokenizer.train(["data/stories.txt"], trainer)
tokenizer.save("data/TinyStories-tokenizer.json")
# -----------------------------


