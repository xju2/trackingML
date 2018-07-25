"""
For each track, assuming there are 20 hits, each hits having 3 coordinates.
So each track has a input dimension of 60.
If less than 20 hits, fill with a large value.

Output: probablity of the input hits to be a real track.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pickle
from sklearn import metrics

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../common')

from utils import tunable_parameters
from utils import device 

from process_data import get_real_tracks
from process_data import get_fake_tracks


def predict_IsGoodTrack(model, input, batch_size=64):
    """
    predict the model of IsGoodTrack
    """
    with torch.no_grad():
        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.expand(batch_size, *input_tensor.size())
        input_tensor = input_tensor.to(device)
        output = model(input_tensor.view(batch_size, -1))
        return output[0]

def get_roc(model, real_tracks, fake_tracks):
    y = []
    scores = []
    for real_track in real_tracks[:1000]:
        output = predict_IsGoodTrack(model, real_track)
        scores.append(output)
        y.append(1)

    for fake_track in fake_tracks[:1000]:
        output = predict_IsGoodTrack(model, fake_track)
        if output is None:
            continue
        scores.append(output)
        y.append(0)


    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0., 1.0)
    plt.ylim(0., 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("roc.png")
    print("scores:", metrics.roc_auc_score(y, scores))

def check():
    # create RNN model
    input_dim = 60
    hidden_dim = 60
    model = IsGoodTrack(input_dim, hidden_dim, batch_size=64, n_lstm_layers=3, device=device)
    model.to(device)
    model.hidden = model.init_hidden()
    print("total parameters:", tunable_parameters(model))
    print(model.hidden[0].type())

    model_name = 'model_isgoodtrack'
    model.load_state_dict(torch.load(model_name))


    real_tracks_out_name = 'reak_tracks_pad_e2.pkl'
    fake_tracks_out_name = 'fake_tracks_pad_e2.pkl'

    event = "input/train_1/event000001001"
    if os.path.exists(real_tracks_out_name):
        real_tracks = pickle.load(open(real_tracks_out_name, 'rb'))
        fake_tracks = pickle.load(open(fake_tracks_out_name, 'rb'))
    else:
        real_tracks = get_real_tracks(event)
        fake_tracks = get_fake_tracks(event, real_tracks.shape[0])
        # save the two dataset
        with open(fake_tracks_out_name, 'wb') as fp:
            pickle.dump(fake_tracks, fp)

        with open(real_tracks_out_name, 'wb') as fp:
            pickle.dump(real_tracks, fp)

    get_roc(model, real_tracks, fake_tracks)

class IsGoodTrack(nn.Module):

    def __init__(self, input_dim=60, hidden_dim=120, batch_size=64, n_lstm_layers=1, device=None):

        super(IsGoodTrack, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), 1)
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
        output = self.fc2(output)
        output = self.sigma(output)
        return output

