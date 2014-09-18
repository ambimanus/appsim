import sys
import os
from datetime import timedelta

# Check if we're headless:
if not os.getenv('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange

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


def plot_aggregated(sc, bd, res=1):
    t = drange(sc.t_start, sc.t_end, timedelta(minutes=res))
    ft = np.array([t[0]] + list(np.repeat(t[1:-1], 2)) + [t[-1]])
    sim_data = np.load(os.path.join(bd, sc.simulation_file))
    P_el = sim_data[:,0,:].sum(0)
    P_el_fill = np.repeat(P_el[:-1], 2)
    T_storage = sim_data[:,2,:]

    fig, ax = plt.subplots(2, sharex=True)
    fig.subplots_adjust(left=0.11, right=0.95, hspace=0.3, top=0.98, bottom=0.2)
    ax[0].set_ylabel('P$_{\mathrm{el}}$ [kW]')
    ymax = max(P_el.max(), 0) / 1000.0
    ymin = min(P_el.min(), 0) / 1000.0
    ax[0].set_ylim(ymin - abs(ymin * 0.1), ymax + abs(ymax * 0.1))
    xspace = (t[-1] - t[-2])
    ax[0].set_xlim(t[0], t[-1] + xspace)
    ax[0].plot_date(t, P_el / 1000.0, fmt='-', color=PRIM, drawstyle='steps-post', lw=0.75)
    # add lw=0.0 due to bug in mpl (will show as hairline in pdf though...)
    ax[0].fill_between(ft, P_el_fill / 1000.0, facecolors=PRIM+(0.5,), edgecolors=EC, lw=0.0)

    ymax = T_storage.max() - 273
    ymin = T_storage.min() - 273
    ax[1].set_ylim(ymin - abs(ymin * 0.01), ymax + abs(ymax * 0.01))
    ax[1].set_ylabel('T$_{\mathrm{storage}}\;[^{\circ}\mathrm{C}]$', labelpad=9)
    for v in T_storage:
        ax[1].plot_date(t, v - 273.0, fmt='-', color=PRIMA, alpha=0.25, lw=0.5)
    ax[1].plot_date(t, T_storage.mean(0) - 273.0, fmt='-', color=PRIMA, alpha=0.75, lw=1.5)

    ax[0].xaxis.get_major_formatter().scaled[1/24.] = '%H:%M'
    ax[-1].set_xlabel('Tageszeit')
    fig.autofmt_xdate()

    return fig


def plot_samples(sc, basedir, idx=None):
    sample_data = np.load(os.path.join(basedir, sc.samples_file))
    if idx is not None:
        sample_data = sample_data[idx].reshape((1,) + sample_data.shape[1:])
    fig, ax = plt.subplots(len(sample_data))
    if len(sample_data) == 1:
        ax = [ax]
    for i, samples in enumerate(sample_data):
        t = np.arange(samples.shape[-1])
        for s in samples:
            ax[i].plot(t, s)

    return fig


def run(sc_file):
    print()
    bd = os.path.dirname(sc_file)
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)
    print(sc.title)

    fig = plot_samples(sc, bd)
    fig.savefig('.'.join((os.path.join(bd, sc.title), str(sc.seed), 'samples', 'pdf')))
    fig.savefig('.'.join((os.path.join(bd, sc.title), str(sc.seed), 'samples', 'png')), dpi=300)

    fig = plot_aggregated(sc, bd)
    fig.savefig('.'.join((os.path.join(bd, sc.title), str(sc.seed), 'pdf')))
    fig.savefig('.'.join((os.path.join(bd, sc.title), str(sc.seed), 'png')), dpi=300)

    if os.getenv('DISPLAY') is not None:
        plt.show()


if __name__ == '__main__':
    for n in sys.argv[1:]:
        if os.path.isdir(n):
            import glob
            for sc_file in glob.glob(os.path.join(n, '*.json')):
                run(sc_file)
        else:
            run(n)
