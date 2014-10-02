import sys
import os
from datetime import timedelta

import numpy as np

import scenario_factory
from analyze import _read_slp, norm


# http://www.javascripter.net/faq/hextorgb.htm
PRIMA = (148/256, 164/256, 182/256)
PRIMB = (101/256, 129/256, 164/256)
PRIM  = ( 31/256,  74/256, 125/256)
PRIMC = ( 41/256,  65/256,  94/256)
PRIMD = ( 10/256,  42/256,  81/256)
EC = (1, 1, 1, 0)
GRAY = (0.5, 0.5, 0.5)
WHITE = (1, 1, 1)


def obj(target, x):
    return np.sum(np.abs(target - x))


def _f(target, x):
    return obj(target, x) / np.sum(np.abs(target))


def p(basedir, fn):
    return os.path.join(basedir, fn)


def resample(d, resolution):
    # resample the innermost axis to 'resolution'
    shape = tuple(d.shape[:-1]) + (int(d.shape[-1]/resolution), resolution)
    return d.reshape(shape).sum(-1)/resolution


def load(f):
    with np.load(f) as npz:
        data = np.array([npz[k] for k in sorted(npz.keys())])
    return data


def stats(fn):
    sc_file = fn
    bd = os.path.dirname(sc_file)
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)
    print(sc.title)

    unctrl = load(p(bd, sc.run_unctrl_datafile))
    pre = load(p(bd, sc.run_pre_datafile))
    block = load(p(bd, sc.run_ctrl_datafile))
    post = load(p(bd, sc.run_post_datafile))
    sched = load(p(bd, sc.sched_file))

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

    # code above is from analyze
    ###########################################################################
    # code below calculates stats

    t_day_start = sc.t_block_start - timedelta(hours=sc.t_block_start.hour,
                                         minutes=sc.t_block_start.minute)
    skip = (t_day_start - sc.t_start).total_seconds() / 60 / 15
    i_block_start = (sc.t_block_start - t_day_start).total_seconds() / 60 / 15
    i_block_end = (sc.t_block_end - t_day_start).total_seconds() / 60 / 15

    P_el_unctrl = unctrl[:,0,skip + i_block_start:skip + i_block_end].sum(0)
    P_el_ctrl = ctrl[:,0,skip + i_block_start:skip + i_block_end].sum(0)
    P_el_sched = ctrl_sched[:,skip + i_block_start:skip + i_block_end].sum(0)

    slp = _read_slp(sc, bd)[skip + i_block_start:skip + i_block_end]

    # Stats
    pairs = [
        # (P_el_sched, P_el_ctrl, 'P_el_sched', 'P_el_ctrl'),
        (P_el_sched, P_el_unctrl, 'P_el_sched', 'P_el_unctrl'),

        (P_el_unctrl, P_el_ctrl, 'P_el_unctrl', 'P_el_ctrl'),
    ]
    st = [sc.title]
    for target, data, tname, dname in pairs:
        diff = obj(target, data)
        perf = max(0, 1 - _f(target, data))
        perf_abs = perf * 100.0
        # if perf_abs > 100.0:
        #     perf_abs = 100 - min(100, max(0, perf_abs - 100))
        print('obj(%s, %s) = %.2f kW (%.2f %%)' % (tname, dname, diff, perf_abs))
        st.append(perf_abs)

    # # SLP
    # diff_ctrl = (P_el_ctrl - P_el_unctrl) / 1000.0
    # slp_ctrl = slp + diff_ctrl
    # slp_range = abs(slp.max() - slp.min())
    # slp_range_ctrl = abs(slp_ctrl.max() - slp_ctrl.min())
    # reduction = (1 - norm(0, slp_range, slp_range_ctrl)) * 100
    # print('slp range = %.2f kW' % slp_range)
    # print('slp range (ctrl) = %.2f kW' % slp_range_ctrl)
    # print('reduction = %.2f %%' % reduction)
    # st.append(reduction)

    # SLP (only schedule)
    diff_sched = (P_el_sched - P_el_unctrl) / 1000.0
    slp_sched = slp + diff_sched
    slp_range = abs(slp.max() - slp.min())
    slp_range_sched = abs(slp_sched.max() - slp_sched.min())
    reduction = (1 - norm(0, slp_range, slp_range_sched)) * 100
    print('slp range = %.2f kW' % slp_range)
    print('slp range (sched) = %.2f kW' % slp_range_sched)
    print('reduction = %.2f %%' % reduction)
    st.append(reduction)


    print()
    return st


def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        pos = rect.get_x()+rect.get_width()/2.
        ax.text(pos, 0.9 * height, '%.1f \\%%' % height, ha='center', va='bottom',
                color=PRIMD, fontsize=5)


def plot_stats(names, target_sched, target_ctrl, target_unctrl, sched_ctrl, sched_unctrl, unctrl_ctrl):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fig = plt.figure(figsize=(6.39, 1.75))
    fig.subplots_adjust(bottom=0.3)
    x = np.arange(len(names))

    ax0 = fig.add_subplot(111)
    ax0.set_xlim(-0.5, x[-1] + 0.5)
    ax0.set_ylim(50, 100)
    ax0.set_ylabel(r"""Planguete [\%]""", fontsize='small')
    ax0.grid(False, which='major', axis='x')
    bars = ax0.bar(x, target_sched, align='center', width=0.5, facecolor=PRIM+(0.5,), edgecolor=EC)
    autolabel(ax0, bars)

    # ax1 = fig.add_subplot(212, sharex=ax0)
    # ax1.set_ylim(50, 100)
    # ax1.set_ylabel(r"""Erbringung [\%]""", fontsize='small')
    # ax1.grid(False, which='major', axis='x')
    # bars = ax1.bar(x, target_ctrl, align='center', width=0.5, facecolor=PRIM+(0.5,), edgecolor=EC)
    # autolabel(ax1, bars)

    # plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.xaxis.set_major_locator(FixedLocator(x))
    ax0.set_xticklabels(names, fontsize='xx-small', rotation=45, rotation_mode='anchor', ha='right')

    return fig


if __name__ == '__main__':
    names = []
    sched_ctrl, sched_unctrl = [], []
    unctrl_ctrl = []
    reduction = []
    for dn in sys.argv[1:]:
        if os.path.isdir(dn):
            st = stats(p(dn, '0.json'))
            for l, d in zip((names, sched_ctrl, sched_unctrl, unctrl_ctrl,
                             reduction),
                            st):
                l.append(d)

    # fig = plot_stats(names, target_sched, target_ctrl, target_unctrl,
    #                  sched_ctrl, sched_unctrl, unctrl_ctrl)
    # fig.savefig(p(os.path.split(dn)[0], 'stats.pdf'))
    # plt.show()
