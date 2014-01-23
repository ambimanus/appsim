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


def plot_aggregated(sc, bd, unctrl, ctrl, ctrl_sched, res=1):
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

    ft = np.array([t[0]] + list(np.repeat(t[1:-1], 2)) + [t[-1]])
    P_el_ctrl_fill = np.repeat(P_el_ctrl[:-1], 2)

    fig, ax = plt.subplots(2, sharex=True)
    fig.subplots_adjust(left=0.11, right=0.95, hspace=0.3, top=0.98, bottom=0.2)
    ax[0].set_ylabel('P$_{\mathrm{el}}$ [kW]')
    ymax = max(P_el_unctrl.max(), P_el_ctrl_fill.max(), P_el_sched.max(), 0) / 1000.0
    ymin = min(P_el_unctrl.min(), P_el_ctrl_fill.min(), P_el_sched.min(), 0) / 1000.0
    ax[0].set_ylim(ymin - abs(ymin * 0.1), ymax + abs(ymax * 0.1))
    # xspace = (t[-1] - t[-2])
    ax[0].set_xlim(t[0], t[24])
    ax[0].axvspan(t[i_block_start], t[i_block_end], fc=GRAY+(0.1,), ec=EC)
    ax[0].axvline(t[0], ls='-', color=GRAY, lw=0.5)
    ax[0].axvline(t[len(t)/2], ls='-', color=GRAY, lw=0.5)
    l_unctrl, = ax[0].plot_date(t, P_el_unctrl / 1000.0, fmt=':', color=PRIMB, drawstyle='steps-post', lw=0.75)
    l_unctrl.set_dashes([1.0, 1.0])
    # add lw=0.0 due to bug in mpl (will show as hairline in pdf though...)
    l_ctrl = ax[0].fill_between(ft, P_el_ctrl_fill / 1000.0, facecolors=PRIM+(0.5,), edgecolors=EC, lw=0.0)
    # Create proxy artist as l_ctrl legend handle
    l_ctrl_proxy = Rectangle((0, 0), 1, 1, fc=PRIM, ec=WHITE, lw=0.0, alpha=0.5)
    l_sched, = ax[0].plot_date(t, P_el_sched / 1000.0, fmt='-', color=PRIM, drawstyle='steps-post', lw=0.75)

    ymax = T_storage_ctrl.max() - 273
    ymin = T_storage_ctrl.min() - 273
    ax[1].set_ylim(ymin - abs(ymin * 0.01), ymax + abs(ymax * 0.01))
    ax[1].set_ylabel('T$_{\mathrm{storage}}\;[^{\circ}\mathrm{C}]$', labelpad=9)
    ax[1].axvspan(t[i_block_start], t[i_block_end], fc=GRAY+(0.1,), ec=EC)
    ax[1].axvline(t[0], ls='-', color=GRAY, lw=0.5)
    ax[1].axvline(t[len(t)/2], ls='-', color=GRAY, lw=0.5)
    for v in T_storage_ctrl:
        ax[1].plot_date(t, v - 273.0, fmt='-', color=PRIMA, alpha=0.25, lw=0.5)
    l_T_med, = ax[1].plot_date(t, T_storage_ctrl.mean(0) - 273.0, fmt='-', color=PRIMA, alpha=0.75, lw=1.5)

    ax[0].xaxis.get_major_formatter().scaled[1/24.] = '%H:%M'
    ax[-1].set_xlabel('Time')
    fig.autofmt_xdate()
    ax[1].legend([l_sched, l_unctrl, l_ctrl_proxy, l_T_med],
                 ['Schedule', 'Uncontrolled', 'Controlled', 'Storage temperatures'],
                 bbox_to_anchor=(0., 1.03, 1., .103), loc=8, ncol=4,
                 handletextpad=0.2, mode='expand', handlelength=3,
                 borderaxespad=0.25, fancybox=False, fontsize='x-small')

    return fig


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

    # plot_samples(sc, bd)
    # plt.show()

    unctrl = np.load(p(bd, sc.run_unctrl_datafile))
    pre = np.load(p(bd, sc.run_pre_datafile))
    block = np.load(p(bd, sc.run_ctrl_datafile))
    post = np.load(p(bd, sc.run_post_datafile))
    sched = np.load(p(bd, sc.sched_file))

    ctrl = np.zeros(unctrl.shape)
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
    minutes = (sc.t_end - sc.t_start).total_seconds() / 60
    assert unctrl.shape[-1] == ctrl.shape[-1] == ctrl_sched.shape[-1]
    shape = unctrl.shape[-1]
    if hasattr(sc, 'slp_file'):
        if minutes == shape:
            print('data is 1-minute resolution, will be resampled by 15')
            res = 15
        elif minutes == shape * 15:
            print('data is 15-minute resolution, all fine')
            res = 1
        else:
            raise RuntimeError('unsupported data resolution: %.2f' % (minutes / shape))
        unctrl = resample(unctrl, res)
        ctrl = resample(ctrl, res)
        ctrl_sched = resample(ctrl_sched, res)
        fig = plot_aggregated_SLP(sc, bd, unctrl, ctrl, ctrl_sched, res=15)
    else:
        if minutes == shape:
            print('data is 1-minute resolution, will be resampled by 60')
            res = 60
        elif minutes == shape * 15:
            print('data is 15-minute resolution, will be resampled by 4')
            res = 4
        elif minutes == shape * 60:
            print('data is 60-minute resolution, all fine')
            res = 1
        else:
            raise RuntimeError('unsupported data resolution: %.2f' % (minutes / shape))
        unctrl = resample(unctrl, res)
        ctrl = resample(ctrl, res)
        ctrl_sched = resample(ctrl_sched, res)
        fig = plot_aggregated(sc, bd, unctrl, ctrl, ctrl_sched, res=60)
    # fig.savefig(p(bd, sc.title) + '.pdf')
    # fig.savefig(p(bd, sc.title) + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    for n in sys.argv[1:]:
        if os.path.isdir(n):
            run(p(n, '0.json'))
        else:
            run(n)
