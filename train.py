import os
import torch
import numpy as np
from model import GPTLanguageModel

# Hyperparameters for a tiny model
batch_size = 32
block_size = 64
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

print(f"Training on {device}...")

vocab_size = 50257 # GPT-2 vocab size

data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'val.bin')

if not os.path.exists(train_path):
    print("train.bin not found! run prepare.py first.")
    exit(1)

train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Convert numpy memmap length to python int
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the checkpoint
os.makedirs('out', exist_ok=True)
torch.save(model.state_dict(), 'out/ckpt.pt')
print("Model saved to out/ckpt.pt!")
