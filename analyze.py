import sys
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange

import scenario_factory


def plot_each_device(sc, unctrl, cntrl):
    t = drange(sc.t_start, sc.t_end, timedelta(minutes=1))
    for d_unctrl, d_ctrl in zip(unctrl, ctrl):
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_ylabel('P$_{el}$ [kW]')
        ymax = max(d_unctrl[0].max(), d_ctrl[0].max()) / 1000.0
        ax[0].set_ylim(-0.01, ymax + (ymax * 0.1))
        ax[0].plot_date(t, d_unctrl[0] / 1000.0, fmt='-', lw=1, label='unctrl')
        ax[0].plot_date(t, d_ctrl[0] / 1000.0, fmt='-', lw=1, label='ctrl')
        leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                            borderaxespad=0.0, fancybox=False)

        ax[1].set_ylabel('T$_{storage}$ [\\textdegree C]')
        ax[1].plot_date(t, d_unctrl[2] - 273.0, fmt='-', lw=1, label='unctrl')
        ax[1].plot_date(t, d_ctrl[2] - 273.0, fmt='-', lw=1, label='ctrl')

        fig.autofmt_xdate()
        for label in leg0.get_texts():
            label.set_fontsize('x-small')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)


def plot_aggregated(sc, unctrl, ctrl, ctrl_sched):
    t = drange(sc.t_start, sc.t_end, timedelta(minutes=1))

    P_el_unctrl = unctrl[:,0,:].sum(0)
    P_el_ctrl = ctrl[:,0,:].sum(0)
    P_el_sched = ctrl_sched.sum(0)

    T_storage_unctrl = unctrl[:,2,:]
    T_storage_ctrl = ctrl[:,2,:]

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].set_ylabel('P$_{el}$ [kW]')
    ymax = max(P_el_unctrl.max(), P_el_ctrl.max()) / 1000.0
    ymin = min(P_el_unctrl.min(), P_el_ctrl.min()) / 1000.0
    ax[0].set_ylim(ymin - (ymin * 0.1), ymax + (ymax * 0.1))
    ax[0].plot_date(t, P_el_unctrl / 1000.0, fmt='-', lw=1, label='unctrl')
    ax[0].plot_date(t, P_el_ctrl / 1000.0, fmt='-', lw=1, label='ctrl')
    ax[0].plot_date(t, P_el_sched / 1000.0, fmt='-', lw=1, label='sched')
    leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    ax[1].set_ylabel('T$_{storage}$ unctrl [\\textdegree C]')
    for v in T_storage_unctrl:
        ax[1].plot_date(t, v - 273.0, fmt='-', color='k', alpha=0.2, lw=1)
    ax[1].plot_date(t, T_storage_unctrl.mean(0) - 273.0, fmt='-', color='k', lw=1)

    ax[2].set_ylabel('T$_{storage}$ ctrl [\\textdegree C]')
    for v in T_storage_ctrl:
        ax[2].plot_date(t, v - 273.0, fmt='-', color='k', alpha=0.2, lw=1)
    ax[2].plot_date(t, T_storage_ctrl.mean(0) - 273.0, fmt='-', color='k', lw=1)

    fig.autofmt_xdate()
    for label in leg0.get_texts():
        label.set_fontsize('x-small')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)


def plot_samples(sc):
    sample_data = np.load(sc.run_pre_samplesfile)
    fig, ax = plt.subplots(len(sample_data))
    for i, samples in enumerate(sample_data):
        t = np.arange(samples.shape[1])
        for s in samples:
            ax[i].plot(t, s)
    plt.show()


if __name__ == '__main__':
    sc_file = sys.argv[1]
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)

    unctrl = np.load(sc.run_unctrl_datafile)
    pre = np.load(sc.run_pre_datafile)
    block = np.load(sc.run_ctrl_datafile)
    post = np.load(sc.run_post_datafile)
    sched = np.load(sc.sched_file)

    ctrl = np.zeros(unctrl.shape)
    idx = 0
    for l in (pre, block, post):
        ctrl[:,:,idx:idx + l.shape[-1]] = l
        idx += l.shape[-1]

    sched = sched.repeat(15, axis=1)
    ctrl_sched = np.zeros((unctrl.shape[0], unctrl.shape[-1]))
    ctrl_sched[:,:pre.shape[-1]] = pre[:,0]
    ctrl_sched[:,pre.shape[-1]:pre.shape[-1] + sched.shape[-1]] = sched
    ctrl_sched[:,pre.shape[-1] + sched.shape[-1]:] = post[:,0]

    # plot_each_device(sc, unctrl, ctrl, sched)
    plot_aggregated(sc, unctrl, ctrl, ctrl_sched)
    # plot_samples(sc)

    plt.show()