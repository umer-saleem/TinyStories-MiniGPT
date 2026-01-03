# TinyStories-MiniGPT: A From-Scratch Character-Level GPT Implementation

A compact GPT-style Transformer model implemented in PyTorch for generating short, creative stories using the TinyStories dataset.

## Overview

This repository presents TinyStories-MiniGPT, a lightweight, from-scratch implementation of a GPT-style character-level language model trained on the TinyStories dataset.

The primary goal of this project is not to achieve state-of-the-art text generation, but to deeply understand the inner workings of transformer-based autoregressive language models.

The project is intentionally designed to be:

- **Minimal** – only the core components are implemented
- **CPU-friendly** – works without GPU, but supports GPU if available
- **Fully readable and modifiable** – ideal for learning and experimentation
- **Educational** – demonstrates step-by-step transformer building and story generation

This implementation shows how **character-level embeddings**, **positional encoding**, **multi-head self-attention**, **residual connections**, and **causal masking** work together to generate text autoregressively.

## Project Objectives

### Key Objective:
To provide a hands-on understanding of the architectural modules in GPT-style transformers and how they contribute to autoregressive text generation.

Specifically, this project focuses on:

- Data preprocessing for character-level modeling
- Tokenization and vocabulary creation
- Implementing self-attention and multi-head attention
- Building feed-forward layers with residual connections
- Positional encoding for sequence modeling
- Causal masking for autoregressive generation
- Training loop with cross-entropy loss
- Sampling and generating coherent stories from a seed prompt

This repository serves as a learning scaffold for anyone who wants to move beyond theory and gain hands-on experience with GPT-like architectures.

## Architecture Overview

The MiniGPT pipeline consists of:

```
The MiniGPT pipeline consists of:

TinyStories Text (~1MB)
        ↓
Data Cleaning & Character Tokenization
        ↓
Character Embeddings + Positional Embeddings
        ↓
Stacked Decoder Blocks:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - LayerNorm & Residual Connections
        ↓
Linear Layer (vocab logits)
        ↓
Softmax → Next Character Prediction
        ↓
Autoregressive Text Generation
```

## Key Components
### Dataset Handling

- Loads TinyStories dataset via HuggingFace datasets library
- Cleans text by removing extra spaces and non-standard characters
- Converts characters into indices for model input
- Splits data into train and test sets (~90% train, 10% test)

**Design Choice:** Character-level modeling simplifies sequence handling and emphasizes understanding transformer behavior without tokenization complexity

### Self-Attention & Multi-Head Attention

- Self-Attention Head: Computes attention scores for each token against all others
- Multi-Head Self-Attention: Uses multiple attention heads in parallel to capture diverse dependencies
- Causal Masking: Prevents the model from attending to future tokens, enabling autoregressive generation

**Educational Insight:** Demonstrates how GPT can learn long-range dependencies without recurrence

### Feed-Forward Blocks with Residuals

- Fully connected layers with ReLU activation
- Residual connections and LayerNorm stabilize training and improve gradient flow

### Positional Embeddings

- Each token is augmented with a learned positional vector
- Enables the model to distinguish token order since self-attention is permutation-invariant

### Transformer Decoder Blocks

- Stacked blocks containing self-attention + feed-forward modules
- Modular design makes it easy to experiment with depth and hidden dimensions

### Autoregressive Text Generation

- Generates text one character at a time conditioned on previously generated characters
- Uses multinomial sampling from softmax probabilities
- Seed text (start_seq) can be customized to influence story style

**Educational Insight:** Shows how causal attention and token embeddings interact for sequence prediction

## Training Strategy

- **Loss:** Cross-entropy loss computed on character predictions
- **Optimizer:** Adam
- **Teacher forcing:** Model predicts next character based on actual previous tokens during training
- **Evaluation:** Periodically generates sample stories to monitor learning progress

**Note:** With small datasets and limited model size, overfitting is expected, which is intentional for learning purposes

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

## Possible Improvements

- Switch to **word-level** or **subword-level tokenization**
- Use **larger embedding dimensions** and deeper models
- Implement **top-k / top-p sampling** for better story generation
- Introduce **dropout** and **learning rate** scheduling
- Train on larger datasets or multiple story corpora

## Why This Repository Matters

This project allows users to:

- Build intuition for transformer-based language models
- Understand attention mechanisms and causal masking
- Gain hands-on experience implementing a GPT-like model from scratch
- Serve as a stepping stone for larger projects like GPT-2, GPT-Neo, or ChatGPT

It is ideal for **students**, **researchers**, and **engineers** who want to understand the fundamentals of transformer-based text generation without relying on large pretrained models.
