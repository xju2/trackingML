# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HitPredictor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=20, output_dim=20,
                n_lstm_layers=1):
        super(HitPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output.view(len(x), -1))
        output = self.dropout(output)
        tag_scores = self.softmax(output)
        return tag_scores


class IsGoodTrack(nn.Module):

    def __init__(self, input_dim=60, hidden_dim=120, batch_size=64, n_lstm_layers=1, device=None):

        super(IsGoodTrack, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, int(input_dim/2))
        self.fc3 = nn.Linear(int(input_dim/2), 1)
        self.fc4 = nn.Linear(9, 1)
        self.sigma = nn.Sigmoid()
        self.device = device

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device),
                torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device)
               )

    def forward(self, x):
        input_size = x.size()
        output, self.hidden = self.lstm(x.view(self.batch_size, 1, -1), self.hidden)
        output = self.fc1(output.view(self.batch_size, self.hidden_dim))
        output = self.sigma(output)
        output = self.fc3(output)
        output = self.sigma(output)
#         output = self.fc4(output)
#         output = self.sigma(output)
        return output
