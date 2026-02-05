import torch
import torch.nn.functional as F
import youtokentome
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Byte pair encoding model
bpe_model = youtokentome.BPE(model = '/Transformer_DL/ssd/transformer data/bpe.model')

#Transformer model
checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar")
model = checkpoint['model'].to(device)
model.eval()

def translate(source_sequence, beam_size = 4, length_norm_coefficient = 0.6):
    """
    Translate a source language sequence into a target language sequence, with beam search decoding
    (A heuristic that keeps finding top K).
    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size (or top K)
    :param length_norm_coefficient: coefficient for normalizing decoded sequences scores for their length
    :return: the best hypothesis sequence and all hypothesis
    """

    with torch.no_grad():
        k = beam_size

        num_hypotheses = min(k, 10)

        vocab_size = bpe_model.vocab_size()

        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(
                [source_sequence],
                output_type = youtokentome.OutputType.ID,
                bos = False,
                eos = False
            )
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)
        else :
            encoder_sequences = source_sequence

        encoder_sequences = encoder_sequences.to(device)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device)

        #Encode
        encoder_sequences = model.encode(encoder_sequences, encoder_sequence_lengths)

        hypotheses = torch.longTensor([[torch.LongTensor(bpe_model.subword_to_id('<BOS>'))]]).to(device) # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(0)]).to(device)
        hypotheses_scores  = torch.zeros(1).to(device)

        complete_hypotheses = list()
        complete_hypotheses_scores = list()

        step = 1

        while True :
            curr_num_hypothesis = hypotheses.size(0)
            decoder_sequences = model.decode(
                decoder_sequences = hypotheses,
                decoder_sequences_length = hypotheses_lengths,
                encoder_sequences = encoder_sequences.repeat(curr_num_hypothesis, 1, 1),
                encoder_sequence_lengths = encoder_sequence_lengths.repeat(curr_num_hypothesis, 1, 1)
            ) # (s, step, vocab_size)

            scores = decoder_sequences[:, -1, :] # score at this step (s, vocab_size)
            scores = F.log_softmax(scores, dim = -1) # (s, vocab_size)
            # Combine with the hypothesis score from last step to all candidates
            scores = hypotheses_scores.unsqueeze(1) + scores #(s, vocab_size)

            # Return the top k scores and their unrolled indices
            top_k_hypotheses_scores, unroll_indices = scores.view(-1).topk(k = k, dim = 0, largest = True, sorted = True)

            prev_word_indices = unroll_indices // vocab_size # (k)
            next_word_indices = unroll_indices  % vocab_size # (k)

            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim = 1)
            # (k, step + 1)

            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # boolean vector of size k

            complete_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow((5 + step) / (5 + 1), length_norm_coefficient)
            complete_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have gained enough hypotheses
            if len(complete_hypotheses) >= num_hypotheses :
                break

            hypotheses = top_k_hypotheses[~complete] # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete] # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)

            if step > 100 :
                break

            step += 1

        # if there is no complete hypothesis then use partial hypothesis
        if len(complete_hypotheses) == 0 :
            complete_hypotheses = hypotheses.tolist()
            complete_hypotheses_scores = hypotheses_scores.tolist()

        all_hypotheses = list()
        for index, hypo in enumerate(bpe_model.decode(complete_hypotheses, ignore_ids=[0, 2, 3])):
           all_hypotheses.append({"hypothesis": hypo, "score": complete_hypotheses_scores[index]})

        best_index = complete_hypotheses_scores.index(max(complete_hypotheses_scores))
        best_hypothesis = all_hypotheses[best_index]["hypothesis"]
        return best_hypothesis, all_hypotheses








