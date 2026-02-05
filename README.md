# Transformer Implementation

A complete PyTorch implementation of the Transformer model from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Overview

This repository contains a clean, well-documented implementation of the Transformer architecture, including:

- **Scaled Dot-Product Attention**: The fundamental attention mechanism
- **Multi-Head Attention**: Parallel attention layers with different learned projections
- **Position-wise Feed-Forward Networks**: Fully connected feed-forward networks applied to each position
- **Positional Encoding**: Sine and cosine functions to inject position information
- **Encoder Stack**: Multiple encoder layers with self-attention and feed-forward sub-layers
- **Decoder Stack**: Multiple decoder layers with masked self-attention, cross-attention, and feed-forward sub-layers
- **Complete Transformer**: End-to-end sequence-to-sequence model

## Architecture

The Transformer follows the encoder-decoder architecture:

```
Input Sequence → Embedding → Positional Encoding → Encoder Stack → 
                                                                   ↓
Output Sequence ← Linear + Softmax ← Decoder Stack ← Positional Encoding ← Embedding
```

### Key Components

1. **Encoder**: Processes the input sequence and produces a continuous representation
2. **Decoder**: Generates the output sequence one token at a time, conditioned on the encoder output
3. **Multi-Head Attention**: Allows the model to attend to different representation subspaces
4. **Residual Connections**: Helps with gradient flow in deep networks
5. **Layer Normalization**: Stabilizes training

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.6+
- PyTorch 1.8+
- NumPy 1.19+

## Usage

### Basic Example

```python
import torch
from transformer import Transformer

# Define model parameters
src_vocab_size = 1000
tgt_vocab_size = 1000
d_model = 512
n_layers = 6
n_heads = 8
d_ff = 2048
max_len = 5000
dropout = 0.1

# Create the model
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    d_ff=d_ff,
    max_len=max_len,
    dropout=dropout
)

# Example input
batch_size = 32
src_seq_len = 10
tgt_seq_len = 10

src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

# Forward pass
output = model(src, tgt)  # Shape: (batch_size, tgt_seq_len, tgt_vocab_size)
```

### With Masking

```python
# Create masks
src_mask = Transformer.create_padding_mask(src, pad_idx=0)
tgt_mask = Transformer.create_look_ahead_mask(tgt_seq_len)

# Forward pass with masks
output = model(src, tgt, src_mask, tgt_mask)
```

### Training Example

Run the provided example script:

```bash
python example.py
```

This script demonstrates:
- Model initialization
- Training loop with dummy data
- Inference with greedy decoding

## Testing

Run the test suite to verify the implementation:

```bash
python test_transformer.py
```

The tests verify:
- Each component's output shapes
- Attention weight normalization
- Mask functionality
- End-to-end model execution

## Model Architecture Details

### Hyperparameters (from the paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Model dimension |
| `n_layers` | 6 | Number of encoder/decoder layers |
| `n_heads` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward network dimension |
| `dropout` | 0.1 | Dropout rate |

### Attention Mechanism

The scaled dot-product attention is computed as:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Multi-head attention allows the model to jointly attend to information from different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Positional Encoding

Since the Transformer contains no recurrence or convolution, positional encodings are added to inject information about token positions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## File Structure

```
.
├── transformer.py          # Main Transformer implementation
├── example.py             # Training and inference example
├── test_transformer.py    # Unit tests
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Implementation Details

- **Xavier Initialization**: All parameters are initialized using Xavier uniform initialization
- **Layer Normalization**: Applied after residual connections
- **Dropout**: Applied to attention weights, residual connections, and positional encodings
- **Masking**: Supports both padding masks and look-ahead masks for decoder

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

This implementation is provided as-is for educational purposes.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes. 
