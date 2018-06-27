#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pickle
from sklearn import metrics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from models import IsGoodTrack

from utils import tunable_parameters
from process_data import get_real_tracks
from process_data import get_fake_tracks


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64

def predict_model(model, input):
    with torch.no_grad():
        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.expand(batch_size, *input_tensor.size())
        input_tensor = input_tensor.to(device)
        # print(input_tensor.size())
        # print(input_tensor.type())
        output = model(input_tensor.view(batch_size, -1))
        return output[0]

def get_roc(model, real_tracks, fake_tracks):
    y = []
    scores = []
    for real_track in real_tracks[:1000]:
        output = predict_model(model, real_track)
        scores.append(output)
        y.append(1)

    for fake_track in fake_tracks[:1000]:
        output = predict_model(model, fake_track)
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

def main():
    # create RNN model
    input_dim = 60
    hidden_dim = 60
    model = IsGoodTrack(input_dim, hidden_dim, batch_size, n_lstm_layers=3, device=device)
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


if __name__ == "__main__":
    main()
