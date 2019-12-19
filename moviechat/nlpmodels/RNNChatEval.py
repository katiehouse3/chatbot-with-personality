from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import dill
import sys
device = 'cpu'
import gzip

PAD_token = 0
SOS_token = 1
EOS_token = 2

class RNNChatEval:
    def __init__(self, encoder, decoder, voc, searcher, device = 'cpu'):
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc
        self.searcher = searcher
        self.device = device
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def indexesFromSentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [self.EOS_token]

    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    def evaluate(self, sentence, max_length=20):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.indexesFromSentence(self.voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)
        # Decode sentence with searcher
        tokens, scores = self.searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words


    def evaluateInput(self, sentence):
        try:
            # Normalize sentence
            sentence = self.normalizeString(sentence)
            # Evaluate sentence
            output_words = self.evaluate(sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD' or x == ".")]
            return ' '.join(output_words)

        except KeyError:
            return "sorry, I couldn't understand that."

if __name__ == '__main__':
    class EncoderRNN(nn.Module):
        def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.embedding = embedding

            # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
            #   because our input size is a word embedding with number of features == hidden_size
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                            dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

        def forward(self, input_seq, input_lengths, hidden=None):
            # Convert word indexes to embeddings
            embedded = self.embedding(input_seq)
            # Pack padded batch of sequences for RNN module
            packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            # Forward pass through GRU
            outputs, hidden = self.gru(packed, hidden)
            # Unpack padding
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
            # Return output and final hidden state
            return outputs, hidden
    class LuongAttnDecoderRNN(nn.Module):
        def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
            super(LuongAttnDecoderRNN, self).__init__()

            # Keep for reference
            self.attn_model = attn_model
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout

            # Define layers
            self.embedding = embedding
            self.embedding_dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)

            self.attn = Attn(attn_model, hidden_size)
    
    class Attn(nn.Module):
        def __init__(self, method, hidden_size):
            super(Attn, self).__init__()
            self.method = method
            if self.method not in ['dot', 'general', 'concat']:
                raise ValueError(self.method, "is not an appropriate attention method.")
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.attn = nn.Linear(self.hidden_size, hidden_size)
            elif self.method == 'concat':
                self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        def dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)

        def general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)

        def concat_score(self, hidden, encoder_output):
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            return torch.sum(self.v * energy, dim=2)

        def forward(self, hidden, encoder_outputs):
            # Calculate the attention weights (energies) based on the given method
            if self.method == 'general':
                attn_energies = self.general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self.concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self.dot_score(hidden, encoder_outputs)

            # Transpose max_length and batch_size dimensions
            attn_energies = attn_energies.t()

            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

    class LuongAttnDecoderRNN(nn.Module):
        def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
            super(LuongAttnDecoderRNN, self).__init__()

            # Keep for reference
            self.attn_model = attn_model
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout

            # Define layers
            self.embedding = embedding
            self.embedding_dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)

            self.attn = Attn(attn_model, hidden_size)

        def forward(self, input_step, last_hidden, encoder_outputs):
            # Note: we run this one step (word) at a time
            # Get embedding of current input word
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)
            # Forward through unidirectional GRU
            rnn_output, hidden = self.gru(embedded, last_hidden)
            # Calculate attention weights from the current GRU output
            attn_weights = self.attn(rnn_output, encoder_outputs)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6
            output = self.out(concat_output)
            output = F.softmax(output, dim=1)
            # Return output and final hidden state
            return output, hidden

    class GreedySearchDecoder(nn.Module):
        def __init__(self, encoder, decoder):
            super(GreedySearchDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, input_seq, input_length, max_length):
            # Forward input through encoder model
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            # Prepare encoder's final hidden layer to be first hidden input to the decoder
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            # Initialize decoder input with SOS_token
            decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
            # Initialize tensors to append decoded words to
            all_tokens = torch.zeros([0], device=device, dtype=torch.long)
            all_scores = torch.zeros([0], device=device)
            # Iteratively decode one word token at a time
            for _ in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input = torch.unsqueeze(decoder_input, 0)
            # Return collections of word tokens and scores
            return all_tokens, all_scores
        

        def evaluateloss(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
            ### Format input sentence as a batch
            # words -> indexes
            indexes_batch = [indexesFromSentence(voc, sentence)]
            # Create lengths tensor
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            # Transpose dimensions of batch to match models' expectations
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            # Use appropriate device
            input_batch = input_batch.to(device)
            lengths = lengths.to(device)
            # Decode sentence with searcher
            loss = searcher(input_batch, lengths, max_length)
            # indexes -> words
            return loss

    # function to evaluate LM perplexity
        def compute_perplexity(pairs_batch, encoder, decoder, max_length=MAX_LENGTH):
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')    
            totalN = 0
            nll = 0 
            for pair in pairs_batch:
              indexes_batch = [indexesFromSentence(voc, pair[0])]
              lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
              input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

              indexes_batch_target = [indexesFromSentence(voc, pair[1])]
              max_target_len = torch.tensor([len(indexes) for indexes in indexes_batch_target])
              target_batch = torch.LongTensor(indexes_batch_target).transpose(0, 1)

              encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
              decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token
              decoder_hidden = encoder_hidden[:decoder.n_layers]

              for i in range(max_target_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # Teacher forcing: next input is current target
                    decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                    #crossEntropy = criterion(decoder_output, target[i])
                    crossEntropy = -torch.log(torch.gather(decoder_output, 1, target_batch[i].view(-1, 1)).squeeze(1))
                    nll += crossEntropy.detach().mean()
                    # Prepare current token to be next decoder input (add a dimension)
                    decoder_input = torch.unsqueeze(decoder_input, 0)
            perplexity = nll /len(pairs_batch)
            return perplexity.data

    
    dill._dill._reverse_typemap['ClassType'] = type
    CURRENT_DIR = os.path.dirname(__file__)
    voc_file = os.path.join(CURRENT_DIR, 'rnn_pkl_files/rnn_voc.pkl')
    encoder_file = os.path.join(
        CURRENT_DIR, 'rnn_pkl_files/rnn_encoder.pkl')
    decoder_file = os.path.join(
        CURRENT_DIR, 'rnn_pkl_files/rnn_decoder.pkl')
    searcher_file = os.path.join(
        CURRENT_DIR, 'rnn_pkl_files/searchable_pkl_unconpressed.pkl')
    voc = dill.load(open(voc_file,"rb"))
    encoder = dill.load(open(encoder_file,"rb"))
    decoder = dill.load(open(decoder_file,"rb"))
    #searcher = dill.load(gzip.open(searcher_file, 'rb'))
    searcher = dill.load(open(searcher_file, 'rb'))

    myEval = RNNChatEval(searcher = searcher, encoder = encoder, voc = voc, decoder = decoder)
    answer = myEval.evaluateInput(sentence=str(sys.argv[1]))
    #dill.settings['recurse'] = True
    #dill.dump(mychat, open('rnn.pkl','wb'))
    print(answer)
    
