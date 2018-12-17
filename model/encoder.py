import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import whatCellType

class EncoderRNN(nn.Module):
    def __init__(self, input_size,  embedding_size, hidden_size, cell_type, depth, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = depth
        self.dropout = dropout
        self.bidirectional = False
        if 'bi' in cell_type:
            self.bidirectional = True
        padding_idx = 3
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.rnn = whatCellType(embedding_size, hidden_size,
                    cell_type, dropout_rate=self.dropout)

    def forward(self, input_tensor, input_lengths, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        input_lens = np.asarray(input_lengths)
        input_seqs = input_tensor.transpose(0,1)
        #batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        if isinstance(hidden, tuple):
            hidden = list(hidden)
            hidden[0] = hidden[0].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden[1] = hidden[1].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden = tuple(hidden)
        else:
            hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        return outputs, hidden