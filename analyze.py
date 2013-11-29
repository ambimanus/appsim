import sys
import os
import csv
from datetime import datetime, timedelta

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


def plot_aggregated(sc, bd, unctrl, ctrl, ctrl_sched):
    res = 1
    if (sc.t_end - sc.t_start).total_seconds() / 60 == unctrl.shape[-1] * 15:
        res = 15
    t_day_start = sc.t_block_start - timedelta(hours=sc.t_block_start.hour,
                                         minutes=sc.t_block_start.minute)
    t = drange(t_day_start, sc.t_end, timedelta(minutes=res))
    skip = (t_day_start - sc.t_start).total_seconds() / 60 / res
    i_block_start = (sc.t_block_start - t_day_start).total_seconds() / 60 / res
    i_block_end = (sc.t_block_end - t_day_start).total_seconds() / 60 / res

    P_el_unctrl = unctrl[:,0,skip:].sum(0)
    P_el_ctrl = ctrl[:,0,skip:].sum(0)
    P_el_sched = ctrl_sched[:,skip:].sum(0)

    T_storage_ctrl = ctrl[:,2,skip:]

    P_el_ctrl_fill_x = np.array([t[0]] + list(t) + [t[-1]])
    P_el_ctrl_fill_y = np.array([0] + list(P_el_ctrl) + [0]) / 1000.0
    # import pdb
    # pdb.set_trace()

    if hasattr(sc, 'slp_file'):
        fig, ax = plt.subplots(3, sharex=True)
    else:
        fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_ylabel('P$_{\mathrm{el}}$ [kW]')
    ymax = max(P_el_unctrl.max(), P_el_ctrl.max()) / 1000.0
    ymin = min(P_el_unctrl.min(), P_el_ctrl.min()) / 1000.0
    ax[0].set_ylim(ymin - (ymin * 0.1), ymax + (ymax * 0.1))
    xspace = (t[-1] - t[-2])
    ax[0].set_xlim(t[0], t[-1] + xspace)
    # ax[0].axvline(t[i_block_start], ls='--', color='0.5')
    # ax[0].axvline(t[i_block_end], ls='--', color='0.5')
    ax[0].axvspan(t[i_block_start], t[i_block_end], fc='0.5', lw=0.0, alpha=0.1)
    ax[0].axvline(t[0], ls='-', color='0.5', lw=0.5)
    ax[0].axvline(t[96], ls='-', color='0.5', lw=0.5)
    ax[0].axvline(t[-1], ls='-', color='0.5', lw=0.5)
    l_unctrl, = ax[0].plot_date(t, P_el_unctrl / 1000.0, fmt='--', color='#6581A4', lw=0.5)
    # l_ctrl, = ax[0].plot_date(t, P_el_ctrl / 1000.0, fmt='-', lw=0.5, label='ctrl')
    l_ctrl, = ax[0].fill(P_el_ctrl_fill_x, P_el_ctrl_fill_y, fc='#1F4A7D', lw=0.0, alpha=0.5)
    l_sched, = ax[0].plot_date(t, P_el_sched / 1000.0, fmt='-', color='#6581A4', lw=0.5)
    # colors = [
    #                 '#348ABD', # blue
    #                 '#7A68A6', # purple
    #                 '#A60628', # red
    #                 '#467821', # green
    #                 '#CF4457', # pink
    #                 '#188487', # turqoise
    #                 '#E24A33', # orange
    #                 '#1F4A7D', # primary
    #                 '#BF9D23', # secondary
    #                 '#BF5B23', # complementary
    #                 '#94A4B6', # primaryA
    #                 '#6581A4', # primaryB
    #                 '#29415E', # primaryC
    #                 '#0A2A51', # primaryD
    #                ][:len(unctrl)]
    # for (c, P_el_unctrl, P_el_ctrl, P_el_sched) in zip(colors, unctrl[:,0,:], ctrl[:,0,:], ctrl_sched):
    #     ax[0].plot_date(t, P_el_unctrl / 1000.0, fmt='-', color=c, lw=1, label='unctrl')
    #     ax[0].plot_date(t, P_el_ctrl / 1000.0, fmt=':', color=c, lw=1, label='ctrl')
    #     ax[0].plot_date(t, P_el_sched / 1000.0, fmt='--x', color=c, lw=1, label='sched')

    ymax = T_storage_ctrl.max() - 273
    ymin = T_storage_ctrl.min() - 273
    ax[1].set_ylim(ymin - (ymin * 0.01), ymax + (ymax * 0.01))
    ax[1].set_ylabel('T$_{\mathrm{storage}}\;[^{\circ}\mathrm{C}]$', labelpad=9)
    ax[1].axvspan(t[i_block_start], t[i_block_end], fc='0.5', lw=0.0, alpha=0.1)
    ax[1].axvline(t[0], ls='-', color='0.5', lw=0.5)
    ax[1].axvline(t[96], ls='-', color='0.5', lw=0.5)
    ax[1].axvline(t[-1], ls='-', color='0.5', lw=0.5)
    for v in T_storage_ctrl:
        ax[1].plot_date(t, v - 273.0, fmt='-', color='#94A4B6', alpha=0.25, lw=0.5)
    l_T_med, = ax[1].plot_date(t, T_storage_ctrl.mean(0) - 273.0, fmt='-', color='#94A4B6', alpha=0.75, lw=1.5)

    if hasattr(sc, 'slp_file'):
        slp = _read_slp(sc, bd)
        diff_ctrl = (P_el_ctrl - P_el_unctrl) / 1000.0
        diff_sched = (P_el_sched - P_el_unctrl) / 1000.0
        ax[2].set_ylabel('P$_{el}$ [kW]')
        ymax = max(slp.max(), (slp + diff_ctrl).max(), (slp + diff_sched).max())
        ymin = min(slp.min(), (slp + diff_ctrl).min(), (slp + diff_sched).min())
        ax[2].set_ylim(ymin - (ymin * 0.1), ymax + (ymax * 0.1))
        ax[2].plot_date(t, slp, fmt='-', lw=1, label='H0')
        ax[2].plot_date(t, slp + diff_ctrl, fmt='-', lw=1, label='H0*')
        ax[2].plot_date(t, slp + diff_sched, fmt='-', lw=1, label='H0* (sched only)')
        leg3 = ax[2].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                            borderaxespad=0.0, fancybox=False)
        for label in leg3.get_texts():
            label.set_fontsize('x-small')

    fig.autofmt_xdate()
    leg0 = ax[1].legend([l_sched, l_unctrl, l_ctrl, l_T_med],
                        ['Einsatzplan', 'ungesteuert', 'gesteuert', 'Speichertemperaturen (Median)'],
                        bbox_to_anchor=(0., 1.03, 1., .103), loc=8, ncol=4,
                        handletextpad=0.2, mode='expand', handlelength=3,
                        borderaxespad=0.25, fancybox=False, fontsize='x-small')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.2, hspace=0.3)

    return fig


def plot_samples(sc, basedir, idx=None):
    sample_data = np.load(p(bd, sc.run_pre_samplesfile))
    if idx is not None:
        sample_data = sample_data[idx].reshape((1,) + sample_data.shape[1:])
    fig, ax = plt.subplots(len(sample_data))
    if len(sample_data) == 1:
        ax = [ax]
    for i, samples in enumerate(sample_data):
        t = np.arange(samples.shape[-1])
        for s in samples:
            ax[i].plot(t, s)


def norm(minimum, maximum, value):
    # return value
    if maximum == minimum:
        return maximum
    return (value - minimum) / (maximum - minimum)


def _read_slp(sc, bd):
    # Read csv data
    slp = []
    found = False
    with open(sc.slp_file, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row:
                continue
            if not found and row[0] == 'Datum':
                found = True
            elif found:
                date = datetime.strptime('_'.join(row[:2]), '%d.%m.%Y_%H:%M:%S')
                if date < sc.t_start:
                    continue
                elif date >= sc.t_end:
                    break
                slp.append(float(row[2].replace(',', '.')))
    slp = np.array(slp)
    # Scale values
    # if hasattr(sc, 'run_unctrl_datafile'):
    #    slp_norm = norm(slp.min(), slp.max(), slp)
    #    unctrl = np.load(p(bd, sc.run_unctrl_datafile)).sum(0) / 1000
    #    slp = slp_norm * (unctrl.max() - unctrl.min()) + unctrl.min()
    MS_day_mean = 13600   # kWh, derived from SmartNord Scenario document
    MS_15_mean = MS_day_mean / 96
    slp = slp / slp.mean() * MS_15_mean

    return slp


def plot_slp(sc, bd):
    slp = _read_slp(sc, bd)

    res = 1
    if (sc.t_end - sc.t_start).total_seconds() / 60 == slp.shape[-1] * 15:
        res = 15
    t = drange(sc.t_start, sc.t_end, timedelta(minutes=res))

    fig, ax = plt.subplots()
    ax.set_ylabel('P$_{el}$ [kW]')
    ymax = max(slp.max(), slp.max())
    ymin = min(slp.min(), slp.min())
    ax.set_ylim(ymin - (ymin * 0.1), ymax + (ymax * 0.1))
    ax.plot_date(t, slp, fmt='-', lw=1, label='H0')
    leg0 = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    fig.autofmt_xdate()
    for label in leg0.get_texts():
        label.set_fontsize('x-small')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)

    return fig


def p(basedir, fn):
    return os.path.join(basedir, fn)


if __name__ == '__main__':
    sc_file = sys.argv[1]
    bd = os.path.dirname(sc_file)
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)

    # plot_samples(sc, bd)
    # plt.show()

    unctrl = np.load(p(bd, sc.run_unctrl_datafile))
    pre = np.load(p(bd, sc.run_pre_datafile))
    block = np.load(p(bd, sc.run_ctrl_datafile))
    post = np.load(p(bd, sc.run_post_datafile))
    sched = np.load(p(bd, sc.sched_file))

    ctrl = np.zeros(unctrl.shape)
    print()
    idx = 0
    for l in (pre, block, post):
        ctrl[:,:,idx:idx + l.shape[-1]] = l
        idx += l.shape[-1]

    if sched.shape[-1] == unctrl.shape[-1] / 15:
        print('Extending schedules shape by factor 15')
        sched = sched.repeat(15, axis=1)
    ctrl_sched = np.zeros((unctrl.shape[0], unctrl.shape[-1]))
    ctrl_sched = np.ma.array(ctrl_sched)
    ctrl_sched[:,:pre.shape[-1]] = np.ma.masked
    ctrl_sched[:,pre.shape[-1]:pre.shape[-1] + sched.shape[-1]] = sched
    ctrl_sched[:,pre.shape[-1] + sched.shape[-1]:] = np.ma.masked

    # plot_each_device(sc, unctrl, ctrl, sched)
    fig = plot_aggregated(sc, bd, unctrl, ctrl, ctrl_sched)
    fig.savefig(p(bd, sc.title) + '.pdf')
    fig.savefig(p(bd, sc.title) + '.png', dpi=300)

    # sc.slp_file = '/home/chh/data/crystal-chp/slp/2010_slp_profile_eon_mitte_ag/H0 - Haushalt.csv'

    # if hasattr(sc, 'slp_file'):
    #     plot_slp(sc, bd)

    plt.show()
