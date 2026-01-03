import torch
from dataset import load_tinystories, get_batch
from model import MiniGPT

# -----------------------------
# Hyperparameters
# -----------------------------
block_size = 128
batch_size = 32
C = 128
head_size = 32
num_heads = 4
n_layers = 4
ff_hidden_mult = 4
learning_rate = 3e-4
epochs = 5
print_every = 100
generate_every = 100
max_new_tokens = 200
start_seq = "Once upon a time"

# -----------------------------
# Load data
# -----------------------------
train_data, test_data, vocab_size, encoder, decoder = load_tinystories()

# -----------------------------
# Initialize model
# -----------------------------
model = MiniGPT(vocab_size, C, num_heads, n_layers, block_size, ff_hidden_mult)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# Training loop
# -----------------------------
step = 0
for epoch in range(epochs):
    for _ in range(len(train_data) // batch_size):
        xb, yb = get_batch(train_data, 'train', block_size, batch_size, train_data, test_data)
        optimizer.zero_grad()
        logits = model(xb)
        B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), yb.view(B*T))
        loss.backward()
        optimizer.step()
        step += 1

        if step % print_every == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        if step % generate_every == 0:
            with torch.no_grad():
                idx = torch.tensor(encoder(start_seq), dtype=torch.long).unsqueeze(0)
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -block_size:]
                    logits = model(idx_cond)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, next_id], dim=1)
                sample_text = decoder(idx[0].tolist())
                print(f"\n--- Generated Sample ---\n{sample_text}\n-----------------------")
