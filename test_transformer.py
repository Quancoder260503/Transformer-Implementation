import torch
import torch.nn as nn
from transformer import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer
)


def test_scaled_dot_product_attention():
    """Test Scaled Dot-Product Attention."""
    print("Testing Scaled Dot-Product Attention...")
    batch_size, n_heads, seq_len, d_k = 2, 4, 10, 64
    
    attention = ScaledDotProductAttention(dropout=0.1)
    attention.eval()  # Set to eval mode to disable dropout for testing
    query = torch.randn(batch_size, n_heads, seq_len, d_k)
    key = torch.randn(batch_size, n_heads, seq_len, d_k)
    value = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    output, weights = attention(query, key, value)
    
    assert output.shape == (batch_size, n_heads, seq_len, d_k), f"Expected shape {(batch_size, n_heads, seq_len, d_k)}, got {output.shape}"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Expected shape {(batch_size, n_heads, seq_len, seq_len)}, got {weights.shape}"
    
    # Check if attention weights sum to 1 (in eval mode, dropout is disabled)
    weight_sum = weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6), "Attention weights should sum to 1"
    
    print("✓ Scaled Dot-Product Attention test passed!")


def test_multi_head_attention():
    """Test Multi-Head Attention."""
    print("Testing Multi-Head Attention...")
    batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8
    
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = mha(x, x, x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Expected shape {(batch_size, n_heads, seq_len, seq_len)}, got {weights.shape}"
    
    print("✓ Multi-Head Attention test passed!")


def test_position_wise_feed_forward():
    """Test Position-wise Feed-Forward Network."""
    print("Testing Position-wise Feed-Forward Network...")
    batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048
    
    ffn = PositionWiseFeedForward(d_model, d_ff, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ffn(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("✓ Position-wise Feed-Forward Network test passed!")


def test_positional_encoding():
    """Test Positional Encoding."""
    print("Testing Positional Encoding...")
    batch_size, seq_len, d_model = 2, 10, 512
    
    pe = PositionalEncoding(d_model, max_len=5000, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("✓ Positional Encoding test passed!")


def test_encoder_layer():
    """Test Encoder Layer."""
    print("Testing Encoder Layer...")
    batch_size, seq_len, d_model, n_heads, d_ff = 2, 10, 512, 8, 2048
    
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = encoder_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("✓ Encoder Layer test passed!")


def test_decoder_layer():
    """Test Decoder Layer."""
    print("Testing Decoder Layer...")
    batch_size, src_seq_len, tgt_seq_len, d_model, n_heads, d_ff = 2, 10, 8, 512, 8, 2048
    
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    output = decoder_layer(x, encoder_output)
    
    assert output.shape == (batch_size, tgt_seq_len, d_model), f"Expected shape {(batch_size, tgt_seq_len, d_model)}, got {output.shape}"
    
    print("✓ Decoder Layer test passed!")


def test_encoder():
    """Test Full Encoder."""
    print("Testing Full Encoder...")
    batch_size, seq_len, vocab_size, d_model, n_layers, n_heads, d_ff = 2, 10, 1000, 512, 6, 8, 2048
    
    encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1)
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("✓ Full Encoder test passed!")


def test_decoder():
    """Test Full Decoder."""
    print("Testing Full Decoder...")
    batch_size, src_seq_len, tgt_seq_len, vocab_size, d_model, n_layers, n_heads, d_ff = 2, 10, 8, 1000, 512, 6, 8, 2048
    
    decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1)
    x = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    output = decoder(x, encoder_output)
    
    assert output.shape == (batch_size, tgt_seq_len, d_model), f"Expected shape {(batch_size, tgt_seq_len, d_model)}, got {output.shape}"
    
    print("✓ Full Decoder test passed!")


def test_transformer():
    """Test Complete Transformer."""
    print("Testing Complete Transformer...")
    batch_size, src_seq_len, tgt_seq_len = 2, 10, 8
    src_vocab_size, tgt_vocab_size = 1000, 1000
    d_model, n_layers, n_heads, d_ff = 512, 6, 8, 2048
    
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    output = transformer(src, tgt)
    
    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), f"Expected shape {(batch_size, tgt_seq_len, tgt_vocab_size)}, got {output.shape}"
    
    print("✓ Complete Transformer test passed!")


def test_transformer_with_masks():
    """Test Transformer with masks."""
    print("Testing Transformer with masks...")
    batch_size, src_seq_len, tgt_seq_len = 2, 10, 8
    src_vocab_size, tgt_vocab_size = 1000, 1000
    d_model, n_layers, n_heads, d_ff = 512, 6, 8, 2048
    
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create masks
    src_mask = Transformer.create_padding_mask(src)
    tgt_mask = Transformer.create_look_ahead_mask(tgt_seq_len)
    
    output = transformer(src, tgt, src_mask, tgt_mask)
    
    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), f"Expected shape {(batch_size, tgt_seq_len, tgt_vocab_size)}, got {output.shape}"
    assert src_mask.shape == (batch_size, 1, 1, src_seq_len), f"Expected src_mask shape {(batch_size, 1, 1, src_seq_len)}, got {src_mask.shape}"
    assert tgt_mask.shape == (1, 1, tgt_seq_len, tgt_seq_len), f"Expected tgt_mask shape {(1, 1, tgt_seq_len, tgt_seq_len)}, got {tgt_mask.shape}"
    
    print("✓ Transformer with masks test passed!")


def test_parameter_count():
    """Test that parameter count is reasonable."""
    print("Testing parameter count...")
    src_vocab_size, tgt_vocab_size = 1000, 1000
    d_model, n_layers, n_heads, d_ff = 512, 6, 8, 2048
    
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff)
    num_params = sum(p.numel() for p in transformer.parameters())
    
    print(f"  Total parameters: {num_params:,}")
    
    # Sanity check: should have millions of parameters for this configuration
    assert num_params > 1_000_000, f"Expected more than 1M parameters, got {num_params:,}"
    
    print("✓ Parameter count test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Transformer Implementation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_scaled_dot_product_attention,
        test_multi_head_attention,
        test_position_wise_feed_forward,
        test_positional_encoding,
        test_encoder_layer,
        test_decoder_layer,
        test_encoder,
        test_decoder,
        test_transformer,
        test_transformer_with_masks,
        test_parameter_count
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            print()
            failed_tests.append(test.__name__)
    
    print("=" * 60)
    if not failed_tests:
        print("All tests passed! ✓")
    else:
        print(f"Failed tests: {', '.join(failed_tests)}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
