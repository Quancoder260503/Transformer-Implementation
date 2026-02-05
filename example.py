import torch
import torch.nn as nn
from transformer import Transformer


def create_example_data(batch_size=32, src_seq_len=10, tgt_seq_len=10, 
                       src_vocab_size=1000, tgt_vocab_size=1000):
    """
    Create example random data for demonstration.
    
    Args:
        batch_size: Number of samples in a batch
        src_seq_len: Length of source sequences
        tgt_seq_len: Length of target sequences
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
    
    Returns:
        src: Source tensor of shape (batch_size, src_seq_len)
        tgt: Target tensor of shape (batch_size, tgt_seq_len)
    """
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    return src, tgt


def train_step(model, src, tgt, optimizer, criterion, device):
    """
    Perform a single training step.
    
    Args:
        model: Transformer model
        src: Source tensor
        tgt: Target tensor
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
    
    Returns:
        loss: Training loss value
    """
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    src = src.to(device)
    tgt = tgt.to(device)
    
    # Create masks
    src_mask = Transformer.create_padding_mask(src).to(device)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    # Create target mask (look-ahead mask combined with padding mask)
    tgt_mask = Transformer.create_look_ahead_mask(tgt_input.size(1)).to(device)
    tgt_padding_mask = Transformer.create_padding_mask(tgt_input).to(device)
    tgt_mask = tgt_mask & tgt_padding_mask
    
    # Forward pass
    output = model(src, tgt_input, src_mask, tgt_mask)
    
    # Calculate loss
    output = output.reshape(-1, output.size(-1))
    tgt_output = tgt_output.reshape(-1)
    loss = criterion(output, tgt_output)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    """
    Main function to demonstrate transformer usage.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    max_len = 5000
    dropout = 0.1
    batch_size = 32
    src_seq_len = 10
    tgt_seq_len = 10
    num_epochs = 5
    learning_rate = 0.0001
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nInitializing Transformer model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Generate random training data
        src, tgt = create_example_data(batch_size, src_seq_len, tgt_seq_len, 
                                      src_vocab_size, tgt_vocab_size)
        
        # Training step
        loss = train_step(model, src, tgt, optimizer, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
    
    print("\nTraining completed!")
    
    # Demonstration: inference on a single example
    print("\nRunning inference on a single example...")
    model.eval()
    with torch.no_grad():
        # Create a single example
        src_example = torch.randint(1, src_vocab_size, (1, src_seq_len)).to(device)
        tgt_start = torch.tensor([[1]]).to(device)  # Start token
        
        # Create source mask
        src_mask = Transformer.create_padding_mask(src_example).to(device)
        
        print(f"Source sequence shape: {src_example.shape}")
        print(f"Target start token shape: {tgt_start.shape}")
        
        # Generate output (greedy decoding for first few tokens)
        max_gen_len = 5
        generated = tgt_start
        
        for _ in range(max_gen_len):
            tgt_mask = Transformer.create_look_ahead_mask(generated.size(1)).to(device)
            output = model(src_example, generated, src_mask, tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        print(f"Generated sequence shape: {generated.shape}")
        print(f"Generated tokens: {generated.squeeze().tolist()}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
