import re
import torch
from datasets import load_dataset

block_size = 128
batch_size = 32
torch.manual_seed(12345)

# Load and preprocess dataset
def load_tinystories(max_chars=1_000_000):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    text = ""
    for t in dataset["text"]:
        text += t + "\n"
        if len(text) > max_chars:
            break
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'"\s\n]', '', text)

    with open("data/tinystories_1mb.txt", "w", encoding="utf-8") as f:
        f.write(text)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encoder = lambda seq: [stoi[ch] for ch in seq]
    decoder = lambda lst: "".join([itos[i] for i in lst])
    
    data = torch.tensor(encoder(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    
    return train_data, test_data, vocab_size, encoder, decoder

# Batch generator
def get_batch(data, split='train', block_size=128, batch_size=32, train_data=None, test_data=None):
    data_split = train_data if split == 'train' else test_data
    idx = torch.randint(0, len(data_split)-block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in idx])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in idx])
    return x, y
