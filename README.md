# TinyStories-MiniGPT

A compact GPT-style Transformer model implemented in PyTorch for generating short, creative stories using the TinyStories dataset.

## Overview

This project demonstrates building a miniature GPT-like language model from scratch, specifically designed for small-scale story generation. It uses the TinyStories dataset and includes:

- **Data preprocessing:** Cleans and prepares TinyStories text for training.
- **Transformer decoder:** Multi-head self-attention, feed-forward blocks, and positional embeddings.
- **Training loop:** Character-level training with cross-entropy loss.
- **Text generation:** Generates stories from a seed prompt using the trained model.

This repository is ideal for learning how transformers work, experimenting with story generation, or fine-tuning small GPT-style models on small datasets.

## Features

- Character-level MiniGPT model with:
  - Multi-Head Self-Attention
  - Feed-Forward Blocks
  - LayerNorm and residual connections
  - Positional embeddings
- Trained on ~1MB of TinyStories text
- Supports generating new story text from a seed prompt
- Simple and modular PyTorch implementation

## Installation
**1.** Clone the repository:
```
git clone https://github.com/umer-saleem/TinyStories-MiniGPT.git
cd TinyStories-MiniGPT
```

**2.** Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

**3.** Install dependencies:
```
pip install torch datasets
```
**Optional:** For GPU support, install the correct PyTorch version with CUDA from PyTorch.

## Usage
### Training

Run the training script:
```
python train.py
```
This will:

**1.** Load and preprocess TinyStories (~1MB of text).

**2.** Train the MiniGPT model for 5 epochs (configurable).

**3.** Print loss and periodically generate sample story text.

### Generating Stories

Use the ```generate.py``` script:
```
from generate import generate, model, encoder, decoder

seed_text = "Once upon a time"
story = generate(model, seed_text, max_new_tokens=200)
print(story)
```

You can modify the ```start_seq``` variable to seed the generation with any prompt.

## Configuration

You can adjust the following hyperparameters in train.py:

- ```block_size``` – Sequence length for training
- ```batch_size``` – Number of sequences per batch
- ```C``` – Embedding dimension
- ```num_heads``` – Number of attention heads
- ```n_layers``` – Number of decoder blocks
- ```ff_hidden_mult``` – Feed-forward hidden size multiplier
- ```learning_rate``` – Optimizer learning rate
- ```epochs``` – Number of training epochs
- ```max_new_tokens``` – Number of tokens to generate

## Dependencies

- Python 3.8+
- PyTorch
- HuggingFace Datasets
- re (standard Python library)
