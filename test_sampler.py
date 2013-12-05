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


def p(basedir, fn):
    return os.path.join(basedir, fn)


def resample(d, resolution):
    # resample the innermost axis to 'resolution'
    shape = tuple(d.shape[:-1]) + (int(d.shape[-1]/resolution), resolution)
    return d.reshape(shape).sum(-1)/resolution


def run(sc_file):
    print()
    bd = os.path.dirname(sc_file)
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)
    print(sc.title)

    sample_data = np.load(p(bd, sc.run_pre_samplesfile))[:,:96]
    unctrl = np.load(p(bd, sc.run_unctrl_datafile))[:,0,:96]

    fig, ax = plt.subplots(len(sample_data))
    if len(sample_data) == 1:
        ax = [ax]
    for i, samples in enumerate(sample_data):
        t = np.arange(samples.shape[-1])
        for s in samples:
            ax[i].plot(t, s)
        ax[i].plot(np.arange(unctrl[i].shape[-1]), unctrl[i], 'k', lw=1.0)

    plt.show()


if __name__ == '__main__':
    for n in sys.argv[1:]:
        if os.path.isdir(n):
            run(p(n, '0.json'))
        else:
            run(n)
