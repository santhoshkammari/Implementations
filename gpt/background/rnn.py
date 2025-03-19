import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity='tanh'
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.wih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.whh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if self.bias:
            self.bih = nn.Parameter(torch.empty(hidden_size))
            self.bhh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_buffer('bih', None)
            self.register_buffer('bhh', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, input, hx=None):
        """
        input: (batch, input_size)
        hx :( batch, hidden_size)
        :param input:
        :param hx:
        :return:
        """
        if hx is None:
            batch_size = input.size(0)
            hx = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype,
                             device=input.device)

        ih = F.linear(input, self.wih, self.bih)
        hh = F.linear(hx, self.whh, self.bhh)

        # add non-linearity
        if self.nonlinearity == 'tanh':
            next_hidden = torch.tanh(ih + hh)
        elif self.nonlinearity == "relu":
            next_hidden = F.relu(ih + hh)
        else:
            raise ValueError(f"Unknow non-linearity")

        return next_hidden


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        bias = True,
        num_layers = 1,
        batch_first = False,
        dropout = 0.0,
        bidirectional = False,
        nonlinearity = 'tanh'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

        num_directions = 2 if bidirectional else 1

        self.rnn_layers = nn.ModuleList()

        for layer in range(self.rnn_layers):
            input_size_layer  = input_size if layer==0 else hidden_size*num_directions



if __name__ == '__main__':
    rnn_layer = RNNLayer(10,20)
    a = torch.randn(size=(2,10))
    rla = rnn_layer(a,torch.ones(size=(2,20)))
    print(a.shape)
    print(rla.shape)
    print(rla)
