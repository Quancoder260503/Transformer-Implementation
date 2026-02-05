import youtokentome
import codecs
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence

from average_checkpoints import source_folder


class SequenceLoader(object):
    """
    An iterator for loading batches of data into the transformer model

    For training :
      Each batch contains token_in_batch target language token,
      target language sequences of the same length to minimize padding and therefore memory usage,
      source language sequences of very similar length to minimize padding, hence memory usage.
      Each batche are also shuffled.

    For validation and testing :
      Each batch contain a single source-target pair
    """

    def __init__(self, data_folder, source_suffix, target_suffix, split, tokens_in_batch):
        """
        :param data_folder: folder containing the source and target language data files.
        :param source_suffix: the filename suffix for source language files
        :param target_suffix: the filename suffix for target language files
        :param split: train, valid or test
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.all_batches = None
        self.current_batch = None
        self.n_batches = None
        self.tokens_in_batch = tokens_in_batch
        self.source_suffix   = source_suffix
        self.target_suffix   = target_suffix

        assert split.lower() in ["train", "valid", "test"], "'Split' must be one of 'train', 'test', 'valid"

        self.split = split.lower()

        self.for_training = self.split == "train"

        self.bpe_model = youtokentome.BPE(model = os.path.join(data_folder, "bpe.model"))

        with codecs.open(os.path.join(data_folder, ".".join([self.split, source_suffix])), "r", "utf-8") as f :
            source_data = f.read().split("\n")[:-1]
        with codecs.open(os.path.join(data_folder, ".".join([self.split, target_suffix])), "r", "utf-8") as f :
            target_data = f.read().split("\n")[:-1]

        assert len(source_data) == len(target_data), "There is a different between the number of sequences in the source and target data"

        source_length = [len(s) for s in self.bpe_model.encode(source_data, bos = False, eos = False)]
        target_length = [len(s) for s in self.bpe_model.encode(target_data, bos = True, eos = True)] # Target sequence has <BOS>, <EOS> tokens

        self.data = list(zip(source_data, target_data, source_length, target_length))

        if self.for_training :
            self.data.sort(key = lambda x : x[3])

        self.create_branches()

    def  create_branches(self):
        if self.for_training :
            chunks = [list(g) for _, g in groupby(self.data, key = lambda x : x[3])] # Group by target sequence length
            self.all_batches = list()
            # Create batches, each with the same target sequence length
            for chunk in chunks :
                chunk.sort(key = lambda x : x[2])
                seqs_per_batch = self.tokens_in_batch // chunk[0][3] # Expected number of sequences per batch
                # Split this chunk of the same target sequence length into batches, to matches the
                # expected number of tokens per batch
                self.all_batches.extend([chunk[i : i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else :
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1 

    def __iter__(self):
        return self

    def __next__(self):
        """
        :return: the next batch, containing :
        source language sequences : a tensor of size (N, encoding_sequence_pad_length)
        target language sequences : a tensor of size (N, decoding_sequence_pad_length)
        true source language lengths : a tensor of size (N, )
        true target language lengths : a tensor of size (N, )
        """

        self.current_batch += 1
        try :
            source_data, target_data, source_length, target_length = self.all_batches[self.current_batch]
        except IndexError :
            raise StopIteration

        source_data = self.bpe_model.encode(source_data, output_type = youtokentome.OutputType.ID, bos = False, eos = False)
        target_data = self.bpe_model.encode(target_data, output_type = youtokentome.OutputType.ID, bos = True,  eos = True)

        # Convert source and target data into padded tensor

        source_data = pad_sequence(sequences = [torch.LongTensor(s) for s in source_data],
                                   batch_first = True,
                                   padding_value = self.bpe_model.subword_to_id('<PAD>'))

        target_data = pad_sequence(sequences = [torch.LongTensor(t) for t in target_data],
                                   batch_first = True,
                                   padding_value = self.bpe_model.subword_to_id('<PAD>'))

        source_lengths = torch.LongTensor(source_length)
        target_lengths = torch.LongTensor(target_length)

        return source_data, target_data, source_length, target_length








