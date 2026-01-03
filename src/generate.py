import torch
from dataset import load_tinystories
from model import MiniGPT

# -----------------------------
# Load data and model
# -----------------------------
_, _, vocab_size, encoder, decoder = load_tinystories()
block_size = 128
C = 128
num_heads = 4
n_layers = 4
ff_hidden_mult = 4

model = MiniGPT(vocab_size, C, num_heads, n_layers, block_size, ff_hidden_mult)
model.eval()

# -----------------------------
# Generation function
# -----------------------------
def generate(model, start_seq="Once upon a time", max_new_tokens=200):
    idx = torch.tensor(encoder(start_seq), dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return decoder(idx[0].tolist())

# -----------------------------
# Example generation
# -----------------------------
start_seq = "Once upon a time"
print(generate(model, start_seq))
