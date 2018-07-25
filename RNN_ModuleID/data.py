#! coding: utf-8
import pickle
import os
import numpy as np
import pandas as pd
import glob

from trackml.dataset import load_event
from utils import make_uID


data_name = os.path.join('output', 'data', 'train_data_uID.pkl.gz')

modules = 18728
def input_target(series):
    tensor = torch.zeros(1, len(series), modules+1)
    for idx, h in enumerate(series):
        tensor[0][idx][h] = 1

    module_idx = series[1:].tolist()
    module_idx.append(modules)
    return tensor, torch.LongTensor(module_idx)


def load_training():
    if os.path.exists(data_name):
        with open(data_name, 'rb') as fp:
            training_df = pickle.load(fp)
    else:
        det = pd.read_csv('input/detectors.csv')
        hits_pd = make_uID(det)

        training_df = []
        train = np.unique([p.split('-')[0] for p in sorted(glob.glob('input/train_1/**'))])
        for ds_name in train:
            hits, truth = load_event(ds_name, parts=['hits', 'truth'])
            hits_with_uID = pd.merge(hits, hits_pd, on=['volume_id', 'layer_id', 'module_id'])
            filtered_truth = truth[ (truth['weight'] > 5e-7 ) & (truth['particle_id'] != 0) ]
            training = hits_with_uID.merge(filtered_truth, on='hit_id')[['uID', 'particle_id']]
            unique_truth = pd.Series(np.unique(training['particle_id']))
            training_df.append( (training, unique_truth) )

        with open(data_name, 'wb') as fp:
            pickle.dump(training_df, fp)
    return training_df


