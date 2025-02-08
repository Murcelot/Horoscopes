from torch import nn
import torch.nn.functional as F
import torch

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, embedding_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          bidirectional=True, batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

    def dot_score(self, hidden, encoder_output):
        return torch.matmul(encoder_output, hidden)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.matmul(energy, hidden)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1)
    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, num_tokens, embedding, embedding_size, hidden_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, num_tokens)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        # embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # return self.out(rnn_output).squeeze(), hidden
        rnn_output = rnn_output.squeeze()
        # hidden = hidden.squeeze()
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output.unsqueeze(-1), encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = torch.matmul(attn_weights.transpose(1, 2), encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        context = context.squeeze()
        concat_input = torch.cat((rnn_output, context), -1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=-1)
        # Return output and final hidden state
        return output, hidden