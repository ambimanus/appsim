# coding=utf-8

import sys
import os
import csv
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange
from matplotlib.patches import Rectangle

import scenario_factory


# http://www.javascripter.net/faq/hextorgb.htm
PRIMA = (148/256, 164/256, 182/256)
PRIMB = (101/256, 129/256, 164/256)
PRIM  = ( 31/256,  74/256, 125/256)
PRIMC = ( 41/256,  65/256,  94/256)
PRIMD = ( 10/256,  42/256,  81/256)
EC = (1, 1, 1, 0)
GRAY = (0.5, 0.5, 0.5)
WHITE = (1, 1, 1)


def load(f):
    with np.load(f) as npz:
        data = np.array([npz[k] for k in sorted(npz.keys())])
    return data


def plot_samples_carpet(sc, basedir, idx=None):
    sample_data = load(p(basedir, sc.run_pre_samplesfile))
    if idx is not None:
        sample_data = sample_data[idx].reshape((1,) + sample_data.shape[1:])
    fig, ax = plt.subplots(len(sample_data))
    if len(sample_data) == 1:
        ax = [ax]
    for i, samples in enumerate(sample_data):
        ax[i].imshow(samples, interpolation='nearest', cmap=plt.get_cmap('binary'), aspect='auto')
        # ax[i].autoscale('x', tight=True)
        ax[i].set_xlabel('Operation schedule interval')
        ax[i].set_ylabel('Sample no.')
        ax[i].grid(False)
    fig.tight_layout()


def p(basedir, fn):
    return os.path.join(basedir, fn)


def run(sc_file):
    print()
    bd = os.path.dirname(sc_file)
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)
    print(sc.title)

    plot_samples_carpet(sc, bd)
    plt.show()


if __name__ == '__main__':
    for n in sys.argv[1:]:
        if os.path.isdir(n):
            run(p(n, '0.json'))
        else:
            run(n)
