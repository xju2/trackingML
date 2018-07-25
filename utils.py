# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_features(df):
    x = df.x.values
    y = df.y.values
    z = df.z.values

    r = np.sqrt(x**2 + y**2 + z**2)

    x2 = x/r
    y2 = y/r

    r2 = np.sqrt(x**2 + y**2)
    z2 = z/r2

    df['x2'] = x2
    df['y2'] = y2
    df['z2'] = z2
    df['r2'] = r2
    df['r'] = r


    # eta, and phi
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    eta = - np.log(np.tan(theta/2.))
    df['eta'] = eta
    df['phi'] = phi
    df['absZ'] = np.abs(z)

    df['z1'] = z/r2

    return df


import matplotlib.ticker as plticker
loc = plticker.MultipleLocator(base=200)
loc0p4 = plticker.MultipleLocator(base=0.4)

def view(df, pID, numb=10, pp=None, pID_name='particle_id'):
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='polar')
    if pp is None:
        if numb < 0:
            pp = pID
        else:
            pp = pID.sample(numb)

    for p in pp:
        if p == 0:
            continue
        data = df[df[pID_name] == p][['r2', 'eta', 'phi', 'z', 'absZ']].sort_values(by=['absZ']).values
        ax1.plot(data[:,3], data[:,0], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax1.scatter(data[:,3], data[:,0], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

        ax2.plot(data[:,3], data[:,1], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax2.scatter(data[:,3], data[:,1], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

        ax3.plot(data[:,3], data[:,2], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax3.scatter(data[:,3], data[:,2], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)


        ax4.plot(data[:,2], np.abs(data[:,3]), '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax4.scatter(data[:,2], np.abs(data[:,3]), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

    ax1.set_xlabel("Z [mm]")
    ax1.set_ylabel("r [mm]")
    ax1.xaxis.set_minor_locator(loc)
    ax1.set_xlim(-3200, 3200)
    ax1.set_ylim(0, 1100)


    ax2.set_xlabel("Z [mm]")
    ax2.set_ylabel("eta")
    ax2.yaxis.set_minor_locator(loc0p4)
    ax2.xaxis.set_minor_locator(loc)
    ax2.set_xlim(-3200, 3200)
    ax2.set_ylim(-5, 5)
    ax2.grid(True)

    ax3.set_xlabel("Z [mm]")
    ax3.set_ylabel("phi")
    ax3.xaxis.set_minor_locator(loc)
    ax3.set_xlim(-3200, 3200)
    ax3.set_ylim(-np.pi, np.pi)

    ax4.grid(True)
    ax4.set_ylim(0, 3200)
    if numb < 12:
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.show()
    return pp, ax1, ax2, ax3, ax4


def make_uID(det):
    data = []
    icount = 0
    volumes = np.unique(det['volume_id'])
    for vol in volumes:
        layers = np.unique(det[det['volume_id'] == vol][['layer_id']])
        for layer in layers:
            modules = np.unique(det[(det['volume_id'] == vol) & (det['layer_id'] == layer)][['module_id']])
            for module in modules:
                data.append([vol, layer, module, icount])
                icount += 1
    return pd.DataFrame(data=data, columns=['volume_id', 'layer_id', 'module_id', 'uID'])


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def tunable_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


import random
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def plot_detector(hits):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    hits.plot.scatter(x='z', y='r2', ax=ax)
    eta = [0.1, 1, 2, 3, 3.2, 4, 4.5]
    # eta = - np.log(np.tan(theta/2.))
    theta = [2 * math.atan(math.exp(-x)) for x in eta]
    xp = np.arange(0, 3200, 200)
    xm = np.arange(-3200, 200, 200)
    for th in theta:
        yp = xp * np.tan(th)
        ym = xm * np.tan(-1 * th)
        ax.plot(xp, yp, '-', alpha=0.5, lw=3)
        ax.plot(xm, ym, '-', alpha=0.5, lw=3)

    ax.set_xlim(-3200, 3200)
    ax.set_ylim(0, 1100)
    ax.xaxis.set_minor_locator(loc)
    plt.tight_layout()
    plt.show()
