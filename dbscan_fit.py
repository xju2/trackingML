#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from trackml.dataset import load_event
# from trackml.randomize import shuffle_hits
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from utils import get_features

from optparse import OptionParser

def recursive_fit_in_eta_phi(dfh, nruns=400, truth=None):
    x = dfh.x.values
    y = dfh.y.values
    z = dfh.z.values

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rt = np.sqrt(x ** 2 + y ** 2)
    theta = np.arccos(z / r)
    phi0 = np.arctan2(y, x)
    eta0 = -np.log(np.tan(theta / 2.0))

    dfh['eta0'] = eta0
    dfh['phi0'] = phi0
    dfh['z1'] = z / rt
    dfh['x2'] = rt / z

    # try a linear dependance of (eta, phi) on Z
    dz0 = -0.00070
    stepdz = 0.00001

    k0_eta = -0.000010
    step_eta = 0.000001

    stepeps = 0.000005
    mm = 1

    scores = []
    epochs = []
    for ii in range(nruns):
        mm = mm * (-1)

        new_ii = ii
        dz = mm * (dz0 + new_ii * stepdz)

        dz_eta = mm * (k0_eta + new_ii * step_eta)

        dfh['phi1'] = phi0 + dz * dfh['z'].values * np.sign(dfh['z'].values)
        dfh['eta1'] = eta0 + dz_eta * dfh['z'].values * np.sign(dfh['z'].values)

        dfh['x1'] = np.cos(dfh['phi1'].values)
        dfh['y1'] = np.sin(dfh['phi1'].values)
        dfh['r1'] = dfh['phi1'].values / dfh['z1']
        dfh['reta1'] = dfh['eta1'].values / dfh['z1']

        dfh['xx1'] = np.cos(dfh['eta1'].values)
        dfh['yy1'] = np.sin(dfh['eta1'].values)

        ss = StandardScaler()
        dfs = ss.fit_transform(dfh[['y1', 'x1', 'z1', 'xx1', 'yy1']].values)
        cx = np.array([1, 1, 0.75, 0.5, 0.5])
        for k in range(dfs.shape[1]):
            dfs[:, k] *= cx[k]

        clusters = DBSCAN(eps=0.0035 + ii * stepeps, min_samples=1, n_jobs=4).fit(dfs).labels_
        if ii == 0:
            dfh['s1'] = clusters
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        else:
            dfh['s2'] = clusters
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
            cond = np.where((dfh['N2'].values > dfh['N1'].values) & (dfh['N2'].values < 20))
            s1 = dfh['s1'].values
            s1[cond] = dfh['s2'].values[cond] + maxs1
            dfh['s1'] = s1
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

        epochs.append(ii)
        sub = dfh[['hit_id', 's1']]
        sub.columns = ['hit_id', 'track_id']
        if truth is not None:
            scores.append(score_event(truth, sub))
        else:
            scores.append(0.)
        del sub

    return dfh['s1'].values, epochs, scores


def fit_event(event):
    hits, = load_event(event, parts=['hits'])
    hits['event_id'] = int(event[-9:])
    hits = get_features(hits)
    hits['particle_id'] = recursive_fit_in_eta_phi(hits)[0]
    return hits[['event_id', 'hit_id', 'particle_id']].copy()


def submit(event, file_name):
    predictions = fit_event(event)
    predictions.to_csv(file_name, index=False)
    
    
def basic_dbscan_fit(hits, eps=0.001, features=['eta', 'phi', 'z2']):
    scl_ = StandardScaler()
    dbscan_ = DBSCAN(eps=eps, min_samples=1, algorithm='auto', n_jobs=1)
    hits_cp = hits.copy()
    hits_cp['track_id'] = dbscan_.fit_predict(scl_.fit_transform(hits_cp[features].values))
    return hits_cp


if __name__ == "__main__":
    usage = "%prog [options] input_event_dir output_csv_dir"
    version = '%prog 0.0.1'
    parser = OptionParser(usage=usage,
                          description="Fit hits with DBScan with recursively changing eta, phi",
                          version=version)
    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        exit(1)

    submit(args[0], args[1])
