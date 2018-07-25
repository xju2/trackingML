#!coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModuleIDBasedRNN(nn.Module):
    """
    Take a model ID and predict the next ID
    """

    def __init__(self, input_dim = 10,
                 hidden_dim=20,
                 output_dim=10,
                 n_lstm_layers=1,
                 batch_size=64,
                 device=None):
        super(ModuleIDBasedRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device),
                torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device))

    def forward(self, x):
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output.contiguous().view(-1, output.size(-1)))

        # output = self.dropout(output)
        tag_scores = self.softmax(output)
        return tag_scores
