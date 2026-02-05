import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention Sublayer.
    """

    def __init__(self, dim_model, num_heads, dim_queries, dim_values, dropout, in_decoder = False):
        """
        :param dim_model: size of the vectors throughout the transformer model  (input and output size of this layer)
        :param num_heads: number of attention heads
        :param dim_queries: size of the query vectors (also the size of the key vectors)
        :param dim_values: size of value vectors
        :param dropout: dropout probability
        :param in_decoder: is it the sublayer of the decoding phase
        """

        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_queries = dim_queries
        self.dim_values  = dim_values
        self.dim_keys    = dim_queries # To allow efficient matrix multiplication, reduce another layer of MLP to match dimension

        self.in_decoder = in_decoder
        # A Linear projection to cast (num_heads sets of) queries from the input query sequences
        self.proj_queries = nn.Linear(dim_model, num_heads * dim_queries)
        # A Linear projection to cast (num heads sets of) keys and values from the input reference sequence
        self.proj_key_values = nn.Linear(dim_model, num_heads * (dim_values + dim_values))
        # A Linear projection to cast (num head sets of)  computed attention weights to the output vector
        self.proj_output = nn.Linear(num_heads * dim_values, dim_model)
        # Softmax Layer
        self.softmax = nn.Softmax(dim = -1)
        # LayerNorm Layer
        self.layer_norm = nn.LayerNorm(dim_model)
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        """
        Forward prop.
        :param query_sequences: the input query sequences, a tensor of shape (batch_size, sequence_pad_length, dim_model)
        :param key_value_sequences: the key_value sequences, a tensor of shape (batch_size, key_value_sequence_pad_length, dim_model)
        :param key_value_sequence_lengths: true length of key_value sequences, a tensor of shape (batch_size, )
        :return: attention-weighted output sequences for the query sequences, a tensor of shape (batch_size, sequence_pad_length, dim_model)
        """
        batch_size = query_sequences.size(0)
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        self_attention = torch.equal(query_sequences, key_value_sequences)

        # Store input for residual connection
        prev_query_sequences = query_sequences.clone()

        query_sequences = self.layer_norm(query_sequences) # (batch_size, query_sequence_pad_length, dim_model)

        # If self_attention, then normalised key value sequences
        # Otherwise, this tensor has been normed in the encoder phase

        if self_attention :
            key_value_sequences = self.layer_norm(key_value_sequences)

        # Project input sequence to query key value
        queries = self.proj_queries(query_sequences) # (batch_size, query_sequence_pad_length, num_heads * d_queries)

        keys, values = self.proj_key_values(key_value_sequence_lengths).split(split_size = self.num_heads * self.dim_keys, dim = -1)
        # (batch_size, query_sequence_pad_length, num_heads * d_key), #(batch_size, query_sequence_pad_length, num_heads * d_value)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.num_heads, self.dim_queries)
        keys    = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.num_heads, self.dim_keys)
        values  = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.num_heads, self.dim_values)

        # Re-arrange the axes such that the last two dimension are sequence length and d_keys/d_values/d_queries,
        # enumerating the new batch size to be batch_size * num_heads, (to apply batch matrix multiplication later)

        # (batch_size * n_heads, query/key_value sequence_pad_length, dim_queries/keys/values)
        queries = queries.permute(0, 2, 1, 3).views(-1, query_sequence_pad_length, self.dim_queries)
        keys    = keys.permute(0, 2, 1, 3).views(-1, key_value_sequence_pad_length, self.dim_keys)
        values  = values.permute(0, 2, 1, 3).views(-1, key_value_sequence_pad_length, self.dim_values)

        # Attention weight learns the relativity of a token in query versus each token in keys
        attention_weights = torch.bmm(queries, keys.permute(0, 2, 1))
        # (batch_size * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)


        # Scaled dot-product
        attention_weights = attention_weights * (1. / math.sqrt(self.dim_keys))

        # Before compute the softmax weights, we should eliminate the effect of padding token


        # (batch_size * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(
            attention_weights
        ).to(device)

        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.num_heads).unsqueeze(0).unsqueeze(0).expand_as(
            attention_weights
        ).to(device)

        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -math.inf)

        # If this is not self-attention in the decoder, keys chronologically ahead of queries
        if self.in_decoder and self_attention :
            # Cannot guest the future if you don't know it
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(device)
            attention_weights = attention_weights.masked_fill(~not_future_mask, -math.inf)

        # compute softmax along the key dimension
        attention_weights = self.softmax(attention_weights)

        # Apply dropout
        dropout = self.dropout_layer(attention_weights)

        # calculate sequences as the weight sum of values based on the softmax weights
        sequences = torch.bmm(attention_weights, values)
        # Tensor shape (batch_size * num_heads, query_sequence_pad_length, dim_values)


        # Un-merged batches and n_heads dimension and restore the original order of axes
        sequences =  (sequences.contiguous().
                     view(batch_size, self.num_heads, query_sequence_pad_length, self.dim_values).
                     permute(0, 2, 1, 3)
        )
        # (batch_size, query_sequence_pad_length, n_heads, dim_values)

        # Merged the last two dimension
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1)


        # Project to the output space
        sequences = self.proj_output(sequences)  # (batch_size, query_sequence_pad_length, dim_model)

        sequences = self.apply(dropout) + prev_query_sequences

        return sequences

class PositionWiseFFN(nn.Module):
    """
    Position-Wise feed forward network layer
    """
    def __init__(self, d_model, d_inner, dropout):
        """
        :param d_model: size of the vectors throughout the transformer model (input and output size)
        :param d_inner: intermediate size or hidden size
        :param dropout: dropout probability
        """

        super(PositionWiseFFN, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_inner)

        self.reLU = nn.ReLU()

        self.fc2 = nn.Linear(d_inner, d_model)

        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward propagation
        :param sequences: input sequences of shape (batch_size, sequence_pad_length, dim_model)
        :return: output sequences of the same shape to the input sequences
        """

        # For residual connection
        prev_sequences = sequences.clone()

        # Normalized the input sequences
        sequences = self.layer_norm(sequences) # (batch_size, sequence_pad_length, dim_model)

        # First Layer
        sequences = self.fc1(sequences)  # (batch_size, sequence_pad_length, dim_inner)
        sequences = self.reLU(sequences)
        sequences = self.apply_dropout(sequences)

        sequences = self.fc2(sequences) # (batch_size, sequence_pad_length, dim_model)

        sequences = self.apply_dropout(sequences) + prev_sequences
        return sequences

class Encoder(nn.Module):
    """
    Encoder architecture
    """
    def __init__(self, vocab_size, positional_encoding, dim_model, num_heads, dim_queries, dim_values, dim_inner,
                 num_layers, dropout):
        """
        :param vocab_size: size of the vocabulary
        :param positional_encoding: positional encoding up to the maximal pad length
        :param dim_model: size of the vectors throughout the transformer model (input and output size)
        :param num_heads: number of heads in the Multi-Head Attention Layer
        :param dim_queries: size of the query vectors (by self-attention mechanism, dim_queries = dim_keys) in the multi-head attention
        :param dim_values: size of the value vectors in the multi-head attention
        :param dim_inner: number of hidden units in the feed-forward layer
        :param num_layers: number of layers (Multihead attention + FFN) in the decoder
        :param dropout: dropout probability
        """
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_queries = dim_queries
        self.dim_values  = dim_values
        self.dim_inner   = dim_inner
        self.num_layers  = num_layers
        self.dropout     = nn.Dropout(dropout)

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, dim_model)

        # Set the positional encoding tensor to be un-update-able
        self.positional_encoding.requires_grad = False

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer() for _ in range(num_layers)])

        # Dropout Layers
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(dim_model)

    def make_encoder_layer(self):
        """
        Create a single layer in the Encoder architecture by combining Multi-Head Attention and FFN layers.
        """
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
                dim_model = self.dim_model,
                num_heads = self.num_heads,
                dim_queries = self.dim_queries,
                dim_values  = self.dim_values,
                dropout     = self.dropout,
                in_decoder  = False
            ),
            PositionWiseFFN(
                d_model = self.dim_model,
                d_inner = self.dim_inner,
                dropout = self.dropout
            )
        ])
        return encoder_layer

    def forward(self, encoder_sequences, encoder_sequences_length):
        """
        Forward propagation
        :param encoder_sequences: the source language sequences, a tensor of size (batch_size, pad_length)
        :param encoder_sequences_length: (true length of encoder sequences)
        :return: encoded source language sequences, a tensor of shape (batch_size, pad_length, dim_model)
        """

        pad_length = encoder_sequences.size(1)

        # Sum vocab embeddings and positional embeddings

        encoder_sequences = (self.embedding(encoder_sequences) * math.sqrt(self.dim_model) +
                             self.positional_encoding[:, : pad_length, :].to(device))# scaled up

        # Dropout
        encoder_sequences = self.apply_dropout(encoder_sequences) # (batch_size, pad_length, dim_model

        for encoder_layer in self.encoder_layers:
            encoder_sequences = encoder_layer[0](
                encoder_sequences,
                encoder_sequences,
                encoder_sequences_length,
            )
            encoder_sequences = encoder_layer[1](
                encoder_sequences
            )

        encoder_sequences = self.layer_norm(encoder_sequences)
        return encoder_sequences



class Decoder(nn.Module):
    """
    Decoder architecture
    """
    def __init__(self, vocab_size, positional_encoding, dim_model, num_heads, dim_queries, dim_values, dim_inner,
                 num_layers, dropout):
        """
        :param vocab_size: size of the vocabulary
        :param positional_encoding: positional encoding up to the maximal pad length
        :param dim_model: size of the vectors throughout the transformer model (input and output size)
        :param num_heads: number of heads in the Multi-Head Attention Layer
        :param dim_queries: size of the query vectors (by self-attention mechanism, dim_queries = dim_keys) in the multi-head attention
        :param dim_values: size of the value vectors in the multi-head attention
        :param dim_inner: number of hidden units in the feed-forward layer
        :param num_layers: number of layers (Multihead attention + FFN) in the decoder
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_queries = dim_queries
        self.dim_values  = dim_values
        self.dim_inner   = dim_inner
        self.num_layers  = num_layers
        self.dropout     = nn.Dropout(dropout)

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, dim_model)

        # Set the positional encoding tensor to be un-update-able
        self.positional_encoding.requires_grad = False

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer() for _ in range(num_layers)])

        # Dropout Layers
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(dim_model)

        # Project back the vocabulary space
        self.fc = nn.Linear(dim_model, vocab_size)

    def make_decoder_layer(self):
        """
        Create a single layer in the Decoder architecture by combining two Multi-Head Attention and a FFN layers.
        """
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
                dim_model = self.dim_model,
                num_heads = self.num_heads,
                dim_queries = self.dim_queries,
                dim_values  = self.dim_values,
                dropout     = self.dropout,
                in_decoder  = True
            ),
            MultiHeadAttention(
                dim_model=self.dim_model,
                num_heads=self.num_heads,
                dim_queries=self.dim_queries,
                dim_values=self.dim_values,
                dropout=self.dropout,
                in_decoder = True
            ),
            PositionWiseFFN(
                d_model = self.dim_model,
                d_inner = self.dim_inner,
                dropout = self.dropout
            )
        ])
        return encoder_layer

    def forward(self, decoder_sequences, decoder_sequences_length, encoder_sequences, encoder_sequences_length):
        """
        Forward propagation
        :param decoder_sequences : the target language sequences, a tensor of size (batch_size, decoder_pad_length)
        :param decoder_sequences_length : (true length of decoder sequences)
        :param encoder_sequences: the source language sequences, a tensor of size (batch_size, encoder_pad_length)
        :param encoder_sequences_length: (true length of encoder sequences)
        :return: encoded source language sequences, a tensor of shape (batch_size, pad_length, dim_model)
        """

        pad_length = encoder_sequences.size(1)

        # Sum vocab embeddings and positional embeddings

        decoder_sequences = (self.embedding(decoder_sequences) * math.sqrt(self.dim_model) +
                             self.positional_encoding[:, : pad_length, :].to(device))# scaled up

        # Dropout
        decoder_sequences = self.apply_dropout(decoder_sequences) # (batch_size, decoder_pad_length, dim_model)

        for decoder_layer in self.decoder_layers:
            # Self-attention
            decoder_sequences = decoder_layer[0](
                decoder_sequences,
                decoder_sequences,
                decoder_sequences_length,
            )

            # Decoder-Encoder Attention
            decoder_sequences = decoder_layer[1](
                decoder_sequences,
                encoder_sequences,
                encoder_sequences_length
            )

            decoder_sequences = decoder_layer[2](
                decoder_sequences
            )

        # apply layer norm
        decoder_sequences = self.layer_norm(decoder_sequences)
        decoder_sequences = self.fc(decoder_sequences) # (batch_size, decoder_pad_length, vocab_size)
        return decoder_sequences



class Transformer(nn.Module):
    """
    Transformer architecture
    """
    def __init__(self,
                 vocab_size,
                 positional_encoding,
                 dim_model = 512,
                 num_heads = 8,
                 dim_queries = 64,
                 dim_values = 64,
                 dim_inner = 2048,
                 num_layers = 6,
                 dropout = 0.1
                 ):
         """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encoding up to the maximal pad length
        :param dim_model: size of the vectors throughout the transformer model (input and output size)
        :param num_heads: number of heads in the Multi-Head Attention Layer
        :param dim_queries: size of the query vectors (by self-attention mechanism, dim_queries = dim_keys) in the multi-head attention
        :param dim_values: size of the value vectors in the multi-head attention
        :param dim_inner: number of hidden units in the feed-forward layer
        :param num_layers: number of layers (Multihead attention + FFN) in the decoder
        :param dropout: dropout probability
         """
         super(Transformer, self).__init__()
         self.vocab_size = vocab_size
         self.positional_encoding = positional_encoding
         self.dim_model = dim_model
         self.num_heads = num_heads
         self.dim_queries = dim_queries
         self.dim_values = dim_values
         self.dim_inner = dim_inner
         self.num_layers = num_layers
         self.dropout = nn.Dropout(dropout)

         self.encoder = Encoder(
             vocab_size = vocab_size,
             positional_encoding = positional_encoding,
             dim_model = dim_model,
             num_heads = num_heads,
             dim_queries = dim_queries,
             dim_values  = dim_values,
             dim_inner   = dim_inner,
             num_layers  = num_layers,
             dropout     = dropout
         )

         self.decoder = Decoder(
             vocab_size=vocab_size,
             positional_encoding=positional_encoding,
             dim_model=dim_model,
             num_heads=num_heads,
             dim_queries=dim_queries,
             dim_values=dim_values,
             dim_inner=dim_inner,
             num_layers=num_layers,
             dropout=dropout
         )

         self.init_weights()

    def init_weights(self):
         """
         Initialize the weights of the model
         """
         for p in self.parameters():
            if p.dim() > 1:
               nn.init.xavier_uniform_(p, gain = 1.0)
         nn.init.normal(self.encoder.embedding.weight, mean = 0.0, std = math.pow(self.dim_model, -0.5))
         self.decoder.embedding.weight = self.encoder.embedding.weight
         self.decoder.fc.weight = self.decoder.embedding.weight
         print("Model initialized.")

    def forward(self, encoder_sequences, encoder_sequences_length, decoder_sequences, decoder_sequences_length):
        """
        Forward propagation
        :param decoder_sequences : the target language sequences, a tensor of size (batch_size, decoder_pad_length)
        :param decoder_sequences_length : (true length of decoder sequences)
        :param encoder_sequences: the source language sequences, a tensor of size (batch_size, encoder_pad_length)
        :param encoder_sequences_length: (true length of encoder sequences)
        :return: encoded source language sequences, a tensor of shape (batch_size, pad_length, dim_model)
        """
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequences_length)

        decoder_sequences = self.decoder(decoder_sequences, decoder_sequences_length, encoder_sequences, encoder_sequences_length)

        return decoder_sequences

class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, eps = 0.1):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        """
        Forward propagation
        :param inputs: decoded target language sequences, a tensor of size (batch_size, decoder_pad_length, vocab_size)
        :param targets: actual target language sequences, a tensor of size (batch_size, decoder_pad_length)
        :param lengths: true length of decoder sequences
        :return: mean label smoothed cross-entropy loss, a scalar
        """

        # Remove Pad Position and Flatten
        inputs, _, _, _ = pack_padded_sequence(
            input = inputs,
            lengths = lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        ) # (sum(lengths), vocab_size)

        targets, _, _, _ = pack_padded_sequence(
            input = targets,
            lengths = lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        ) # (sum(lengths))

        # (sum(lengths), vocab_size)
        target_vector = torch.zeros_like(inputs).scatter(dim = 1, index = targets.unsqueeze(1), value = 1.0).to(device)
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(1)

        loss = (-1 * target_vector * F.log_softmax(inputs, dim = -1)).sum(dim = 1) # (sum(lengths), )

        loss = torch.mean(loss)
        return loss






