import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.util import whatCellType


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        padding_idx = 3
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx
                                      )
        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size, hidden_size, cell_type, dropout_rate=dropout)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1,B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        #output = embedded
        output = F.relu(embedded)

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)

        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)

        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)  # [T,B,H] -> [B,T,H]
        attn_energies = self.score(H,encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        cat = torch.cat([hidden, encoder_outputs], 2)
        energy = torch.tanh(self.attn(cat)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class SeqAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout_p=0.1, max_length=30):
        super(SeqAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.output_size = output_size
        self.n_layers = 1
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size + hidden_size, hidden_size, cell_type, dropout_rate=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, embedding_size)

        # attention
        self.method = 'concat'
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, input, hidden, encoder_outputs):
        if isinstance(hidden, tuple):
            h_t = hidden[0]
        else:
            h_t = hidden
        encoder_outputs = encoder_outputs.transpose(0, 1)
        embedded = self.embedding(input)  # .view(1, 1, -1)
        # embedded = F.dropout(embedded, self.dropout_p)

        # SCORE 3
        max_len = encoder_outputs.size(1)
        h_t = h_t.transpose(0, 1)  # [1,B,D] -> [B,1,D]
        h_t = h_t.repeat(1, max_len, 1)  # [B,1,D]  -> [B,T,D]
        energy = self.attn(torch.cat((h_t, encoder_outputs), 2))  # [B,T,2D] -> [B,T,D]
        energy = torch.tanh(energy)
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        attn_weights = F.softmax(energy, dim=2)  # [B,1,T]

        # getting context
        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H]

        # context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) #[B,1,H]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        rnn_input = rnn_input.transpose(0, 1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden  # , attn_weights
