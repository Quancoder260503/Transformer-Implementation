import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism as described in "Attention is All You Need".
    
    The attention function computes the dot products of the query with all keys,
    divides each by sqrt(d_k), and applies a softmax function to obtain weights on values.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, n_heads, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len) or broadcastable
        
        Returns:
            output: Attention output of shape (batch_size, n_heads, seq_len, d_v)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need".
    
    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Multi-head attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in "Attention is All You Need".
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    This consists of two linear transformations with a ReLU activation in between.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward network
            dropout: Dropout rate
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in "Attention is All You Need".
    
    Since the model contains no recurrence and no convolution, positional encodings
    are added to give the model information about the relative or absolute position
    of tokens in the sequence.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of the module state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Input with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer as described in "Attention is All You Need".
    
    Each encoder layer consists of:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    Each sub-layer has a residual connection and layer normalization.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer as described in "Attention is All You Need".
    
    Each decoder layer consists of:
    1. Masked multi-head self-attention mechanism
    2. Multi-head cross-attention mechanism (attending to encoder output)
    3. Position-wise feed-forward network
    Each sub-layer has a residual connection and layer normalization.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor (for masking future positions)
        
        Returns:
            output: Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual connection and layer normalization
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    Full Encoder stack as described in "Attention is All You Need".
    
    The encoder consists of N identical encoder layers.
    """
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            n_layers: Number of encoder layers
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            mask: Optional mask tensor
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Embed and scale by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Decoder(nn.Module):
    """
    Full Decoder stack as described in "Attention is All You Need".
    
    The decoder consists of N identical decoder layers.
    """
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len) containing token indices
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
        
        Returns:
            output: Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embed and scale by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model as described in "Attention is All You Need".
    
    The Transformer consists of an encoder and a decoder, along with final linear
    and softmax layers for generating output probabilities.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Dimension of the model (default: 512)
            n_layers: Number of encoder/decoder layers (default: 6)
            n_heads: Number of attention heads (default: 8)
            d_ff: Dimension of the feed-forward network (default: 2048)
            max_len: Maximum sequence length (default: 5000)
            dropout: Dropout rate (default: 0.1)
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters with Xavier uniform initialization
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask tensor
            tgt_mask: Optional target mask tensor
        
        Returns:
            output: Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(decoder_output)
        return output
    
    @staticmethod
    def create_padding_mask(seq, pad_idx=0):
        """
        Create padding mask for sequences.
        
        Args:
            seq: Input sequence tensor of shape (batch_size, seq_len)
            pad_idx: Padding token index (default: 0)
        
        Returns:
            mask: Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    @staticmethod
    def create_look_ahead_mask(size):
        """
        Create look-ahead mask for decoder self-attention to prevent attending to future positions.
        
        Args:
            size: Sequence length
        
        Returns:
            mask: Look-ahead mask of shape (1, 1, size, size)
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = ~mask
        return mask.unsqueeze(0).unsqueeze(0)
