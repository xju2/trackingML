"""
Scripts to prepare the dataset for training and testing
"""

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import torch

import pandas as pd
import numpy as np

import glob
import os
import logging
import pickle

# import from load modules
from utils import make_uID
from utils import random_choice
from utils import get_features


def get_real_tracks(event):
    hits, cells, particles, truth = load_event(event)
    hits_truth = pd.merge(hits, truth, on=['hit_id'])
    pIDs = np.unique(hits_truth['particle_id'])
    track_list = []
    for pID in pIDs:
        if pID == 0:
            continue
        this_track = hits_truth[hits_truth['particle_id'] == pID][['x', 'y', 'z']].values
        this_track = np.pad(this_track, ((0, 20), (0, 0)), 'constant', constant_values=(4, 99999))[:20, :]
        track_list.append(this_track)
    return np.array(track_list)


def normalize_r_phi_z():
    track_list_raw = pickle.load(open('ten_hists.npy', 'rb'))

    mean_r, sigma_r = 913.681763, 692.430542
    mean_phi, sigma_phi = 0.009939, 1.823752
    mean_z, sigma_z = -2.315056, 1061.912476
    for event in track_list_raw:
        for tracks in event:
            tracks[:, 0] = (tracks[:, 0] - mean_r)/sigma_r
            tracks[:, 1] = (tracks[:, 1] - mean_phi)/sigma_phi
            tracks[:, 2] = (tracks[:, 2] + mean_z)/sigma_z

    track_arrays = np.concatenate([np.array(x) for x in track_list_raw])
    print("total number of tracks: {}, with {} hits".format(
        track_arrays.shape[0], track_arrays.shape[1]))

    outname = 'ten_hists_normed.npy'
    with open(outname, 'wb') as fp:
        pickle.dump(track_arrays, fp)
    print("tracks written to: ", outname)

import random
def get_fake_tracks(event, n_sample=5000):
    hits, = load_event(event, parts=['hits'])
    fake_tracks = []
    for idx in range(n_sample):
        rand_hits = random.randint(0, 20)
        data = hits.sample(rand_hits)[['x', 'y', 'z']].values
        data = np.pad(data, ((0, 20), (0, 0)), 'constant', constant_values=(4, 99999))[:20, :]
        fake_tracks.append(data)
    return np.array(fake_tracks)


def random_inputs(real_tracks, fake_tracks, batch_size):
    nsample = int(batch_size/2)
    start = random.randint(0, real_tracks.shape[0]-nsample)
    tracks = np.concatenate((real_tracks[start: start+nsample],
                             fake_tracks[start: start+nsample]), axis=0)

    target = [1]*nsample + [0]*nsample
    return tracks, np.array(target)


def filter_truth(truth):
    return truth[ (truth['weight'] > 5e-7 ) & (truth['particle_id'] != 0) ]


def training_data_with_eta_cut(train_dir='input/train_1', event_prefix="event000001000", eta_cut=3.2):
    hits, cells, particles, truth = load_event(os.path.join(train_dir, event_prefix))

    hits_features = get_features(hits)
    # high_eta_hits = hits_features[(hits_features['eta'] > eta_cut) | (hits_features['eta'] < -1 * eta_cut)]
    high_eta_hits = hits_features[(hits_features['eta'] > eta_cut)]
    uID_for_higheta = make_uID(high_eta_hits)
    high_eta_hits_uID = pd.merge(high_eta_hits, uID_for_higheta, on=['volume_id', 'layer_id', 'module_id'])
    train_data_higheta = high_eta_hits_uID.merge(filter_truth(truth), on='hit_id')[['uID', 'particle_id']]
    return train_data_higheta, uID_for_higheta
    #return train_data_higheta, uID_for_higheta.shape[0]


class data_uID(object):
    """
    Each hit is represented by a unique ID,
    derived from its associated volume ID, layer ID and module ID
    """
    def __init__(self, input_path='input'):
        self.event_list = {}
        self.input_path = input_path
        # self.det = pd.read_csv(os.path.join(input_path, 'detectors.csv'))
        # self.det_uID = make_uID(self.det)
        # self.modules = self.det_uID.shape[0]
        # self.data_types = dict([('all', self.read_event),
        #                         ('high_eta', self.read_event_with_eta_cut)
        #                         ])

    def load_training(self, path, eta_cut=3.2):
        if 'event' in path:
            hits_with_truth, particles = self.read_event(path, eta_cut)
            df_uID = make_uID(hits_with_truth)
            self.modules = df_uID.shape[0]
            hits_with_truth = pd.merge(hits_with_truth, df_uID, on=['volume_id', 'layer_id', 'module_id'])
            self.event_list[path] = (hits_with_truth, particles)
        else:
            train = np.unique([p.split('-')[0] for p in sorted(glob.glob(path + '/**'))])
            data_list = [self.read_event(ee) for ee in train]
            det = pd.read_csv(os.path.join(self.input_path, 'detectors.csv'))
            df_uID = make_uID(det)
            self.modules = df_uID.shape[0]
            for event in train:
                hits_with_truth, particles = self.read_event(event, eta_cut)
                hits_with_truth = pd.merge(hits_with_truth, df_uID, on=['volume_id', 'layer_id', 'module_id'])
                self.event_list[event] = (hits_with_truth, particles)

            logging.info('total events: {}'.format(len(self.event_list.keys())))
            logging.info('total truth particles: {:e}'.format(sum([y.shape[0] for x, y in self.event_list.values()])))


    def read_event(self, path, eta_cut=3.2):
        hits, cells, particles, truth = load_event(path)

        hits_features = get_features(hits)
        # apply the eta cuts on hits
        hits_features = hits_features[(hits_features['eta'] > eta_cut) | (hits_features['eta'] < -1 * eta_cut)]

        hits_with_truth = hits_features.merge(filter_truth(truth), on='hit_id')
        particles = pd.Series(np.unique(hits_with_truth['particle_id']))

        return hits_with_truth, particles

    def get_a_track(self):
        """ randomly pick a particle,
        return it's associated list of hits"""
        event_key = random_choice(list(self.event_list.keys()))
        train, truth = self.event_list.get(event_key)
        pID = truth.sample(1).values
        hits = train[train['particle_id'] == pID[0]]['uID'].values
        return hits

    def random_input_target(self):
        """
        randomly chose one track from track pool
        """
        hits = self.get_a_track()
        input_ = self.input_hits(hits)
        target_ = self.target_hits(hits)
        return input_, target_

    def input_hits(self, series):
        tensor = torch.zeros(len(series), 1, self.modules+1)
        for idx, h in enumerate(series):
            tensor[idx][0][h] = 1
        return tensor

    def target_hits(self, series):
            module_idx = series[1:].tolist()
            module_idx.append(self.modules)
            return torch.LongTensor(module_idx)
