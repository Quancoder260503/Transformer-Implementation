import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import time
from model import Transformer, LabelSmoothedCrossEntropy
from dataloader import SequenceLoader
from utils import *

data_source_folder = '/Transformer_DL/ssd/transformer data'

# Model parameters
dim_model = 512  # size of vectors throughout the transformer model
num_heads = 8  # number of heads in the multi-head attention
dim_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
dim_values = 64  # size of value vectors in the multi-head attention
dim_inner = 2048  # an intermediate size in the position-wise FFN
num_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
positional_encoding = get_positional_encoding(d_model = dim_model, max_length = 160)  # positional encodings up to the maximum possible pad-length

# Learning parameters
checkpoint = 'transformer_checkpoint.pth.tar'
tokens_in_batch = 2000
batches_per_step = 25000 // tokens_in_batch  # perform a training step, i.e. update parameters, once every so many batches
print_frequency = 20
n_steps = 100000
warmup_steps = 8000
step = 1
lr = get_learning_rate(step = step, d_model= dim_model, warmup_steps = warmup_steps)

betas = (0.9, 0.98)  # beta coefficients in the Adam optimizer
epsilon = 1e-9  # epsilon term in the Adam optimizer
label_smoothing = 0.1  # label smoothing co-efficient in the Cross Entropy loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = False

def train(train_loader, model, criterion, optimizer, epoch, step):
    """
    An epoch training
    :param train_loader: loader for training data
    :param model: model
    :param criterion: label-smoothed cross entropy loss
    :param optimizer: optimizer
    :param epoch: epoch number
    :param step: step number
    """
    # Training mode allows dropout
    model.train()

    data_time = AverageTracker()
    step_time = AverageTracker()
    losses = AverageTracker()

    start_data_time = time.time()
    start_step_time = time.time()

    for index, (
        source_sequences,
        target_sequences,
        source_sequence_lengths,
        target_sequence_lengths
    ) in enumerate(train_loader):

        source_sequences = source_sequences.to(device)  # (batch_size, encoder_pad_length)
        target_sequences = target_sequences.to(device)  # (batch_size, decoder_pad_length)
        source_sequence_lengths = source_sequence_lengths.to(device)  # (batch_size)
        target_sequence_lengths = target_sequence_lengths.to(device)  # (batch_size)

        data_time.update(time.time() - start_data_time)

        predicted_sequences = model(
            source_sequences,
            target_sequences,
            source_sequence_lengths,
            target_sequence_lengths
        )  # (N, sequence_pad_length, vocab_size)

        # Ignore <BOS> for target sequences
        loss = criterion(
            inputs=predicted_sequences,
            targets=target_sequences[:, 1:],
            lengths=target_sequence_lengths - 1
        )

        # Backward propagation
        (loss / batches_per_step).backward()

        # Track loss
        losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

        # Update model after batches_per_step batches
        if (index + 1) % batches_per_step == 0:
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            change_lr(
                optimizer,
                new_lr=get_learning_rate(
                    step=step,
                    d_model=dim_model,
                    warmup_steps=warmup_steps
                )
            )

            step_time.update(time.time() - start_step_time)

            if step % print_frequency == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {index + 1}/{train_loader.n_batches} | "
                    f"Step {step}/{n_steps} | "
                    f"Data {data_time.val:.3f}s ({data_time.avg:.3f}s avg) | "
                    f"Step {step_time.val:.3f}s ({step_time.avg:.3f}s avg) | "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f} avg)"
                )

            # Reset step time
            start_step_time = time.time()

            # Save checkpoints for averaging in final epochs
            if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    prefix_dir_name ='step' + str(step) + "_"
                )
        # Reset data time
        start_data_time = time.time()



def validate(val_loader, model, criterion):
   """
   Epoch validation
   :param val_loader: loader for validation data
   :param model: transformer model
   :param criterion: label-smoothed cross entropy loss
   """

   model.eval() # Switch to eval mode
   with torch.no_grad():
       losses = AverageTracker()
       for index, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
               val_loader):
           source_sequence = source_sequence.to(device)  # (1, source_sequence_length)
           target_sequence = target_sequence.to(device)  # (1, target_sequence_length)
           source_sequence_length = source_sequence_length.to(device) # (1)
           target_sequence_length = target_sequence_length.to(device) # (1)


           predicted_sequences = model(
               source_sequence,
               target_sequence,
               source_sequence_length,
               target_sequence_length
           )  # (1, sequence_pad_length, vocab_size)

           # Ignore <BOS> for target sequences
           loss = criterion(
               inputs=predicted_sequences,
               targets=target_sequence[:, 1:],
               lengths=target_sequence_length - 1  # scalar
           )
           # Keep track of loss
           losses.update(loss.item(), (target_sequence_length - 1).sum().item())
   print(f"\nValidation Loss : {losses.avg:.3f}")

def main():
    """
    Train and evaluate
    """
    global checkpoint, step, start_epoch, epoch, epochs
    train_loader = SequenceLoader(
        data_folder = data_source_folder,
        source_suffix = "en",
        target_suffix = "de",
        split = "train",
        tokens_in_batch = tokens_in_batch
    )

    val_loader = SequenceLoader(
        data_folder = data_source_folder,
        source_suffix = "en",
        target_suffix = "de",
        split = "val",
        tokens_in_batch = tokens_in_batch
    )

    if checkpoint is None:
        model = Transformer(
            vocab_size = train_loader.bpe_model.vocab_size(),
            positional_encoding = positional_encoding,
            dim_model = dim_model,
            num_heads = num_heads,
            dim_queries = dim_queries,
            dim_values  = dim_values,
            dim_inner = dim_inner,
            num_layers = num_layers,
            dropout    = dropout
        )
        optimizer = torch.optim.Adam(
           [p for p in model.parameters() if p.requires_grad],
            lr = lr,
            betas = betas,
            eps = epsilon
        )

    else :
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f"\nLoaded checkpoint from epoch {start_epoch}")
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Loss function
    criterion = LabelSmoothedCrossEntropy(eps = label_smoothing)

    # Move model to devices
    model = model.to(device)
    criterion = criterion.to(device)

    epochs = (n_steps // (train_loader.n_batches // batches_per_step)) + 1

    for epoch in range(start_epoch, epochs):
        step = epoch * train_loader.n_batches // batches_per_step

        train_loader.create_branches()
        train(
            train_loader = train_loader,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            epoch = epoch,
            step = step
        )
        val_loader.create_branches()
        validate(
            val_loader = val_loader,
            model = model,
            criterion = criterion
        )

        save_checkpoint(epoch, model, optimizer)

if __name__ == "__main__":
    main()

