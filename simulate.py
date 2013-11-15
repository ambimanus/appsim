import sys
import time
import random
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import drange

import device_factory
from progress import PBar


def simulate(device, start, end, progress, newline=False):
    # Datenfelder
    headers = ['P_el', 'P_th', 'T', 'T_env']
    data = {h: np.empty((end - start,)) for h in headers}

    # Simulation
    for now in range(start, end):
        device.step(now)
        i = now - start
        data['P_el'][i] = device.components.engine.P_el
        data['P_th'][i] = device.components.engine.P_th
        data['T'][i] = device.components.storage.T
        data['T_env'][i] = device.components.heatsink.T_env
        progress.update()

    progress.flush()
    if newline:
        print()
    # print('heatsink.annual_demand = %.2f' % device.components.heatsink.annual_demand)
    # print('heatsink.in_heat = %.2f' % (sum(data['in_heat'])/60.0))

    return data


def create_sample(device, sample_size, t_start, t_end, progress, density=0.1):
    device = device.copy()
    device.step(t_start)
    device.components.sampler.setpoint_density = density
    d = (t_end - t_start) / 15
    sample = np.array(device.components.sampler.sample(sample_size, duration=d))
    # Add noise to prevent breaking the SVDD model due to linear dependencies
    noise = np.abs(np.random.normal(scale=0.0001, size=(sample_size, d)))
    progress.update(progress.currval + sample_size)

    return (sample / 1000) + noise


def plot_sim(t, device, data):
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot_date(t, data['P_el'] / 1000.0, fmt='-', lw=1, label='P$_{el}$ [kW]')
    ax[0].plot_date(t, data['P_th'] / 1000.0, fmt='-', lw=1, label='P$_{th}$ [kW]')
    leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    ax[1].plot_date(t, data['T'] - 273, fmt='-', lw=1, label='T [\\textdegree C]')
    ax[1].plot_date(t, data['T_env'], fmt='-', lw=1, label='T$_{env}$')
    T_min = np.array([device.components.engine.T_min for x in t])
    T_max = np.array([device.components.engine.T_max for x in t])
    ax[1].plot_date(t, T_min - 273, fmt='k-', lw=1, label='T$_{min}$')
    ax[1].plot_date(t, T_max - 273, fmt='k-', lw=1, label='T$_{max}$')
    leg1 = ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    for label in leg0.get_texts() + leg1.get_texts():
        label.set_fontsize('x-small')
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    plt.show()


def plot_sample(t, sample):
    fig, ax = plt.subplots()
    ax.set_ylabel('P$_{el}$ [kW]')

    for s in sample:
        ax.plot_date(t, s, fmt='-', lw=1)

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    plt.show()


def resample(d, resolution):
    return (d.reshape(d.shape[0]/resolution, resolution).sum(1)/resolution)


def run(sc):
    p_sim = PBar(len(sc.devices) * (sc.i_block_start - sc.i_pre)).start()
    p_sam = PBar(len(sc.devices) * sc.sample_size).start()

    sim_data = []
    sample_data = []
    for d in sc.devices:
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, p_sim)
        # Simulation
        sim_data.append(simulate(d, sc.i_start, sc.i_block_start, p_sim))
        # Sampling
        sample_data.append(create_sample(d, sc.sample_size, sc.i_block_start,
                                         sc.i_block_end, p_sam))
    return sim_data, sample_data


if __name__ == '__main__':
    sim, sam, export = False, False, False
    if len(sys.argv) == 1 or 'simulate' in sys.argv[1:]:
        sim = True
    if 'sample' in sys.argv[1:]:
        sam = True
    if 'export' in sys.argv[1:]:
        export = True
    try:
        sample_size = int(sys.argv[sys.argv.index('sample') + 1])
    except:
        sample_size = 2
    try:
        devices = int(sys.argv[sys.argv.index('export') + 1])
    except:
        devices = 1

    # Simulationszeit
    start, end = datetime(2010, 4, 1), datetime(2010, 4, 8)
    istart = int(time.mktime(start.timetuple()) // 60)
    iend = int(time.mktime(end.timetuple()) // 60)

    p_sim = PBar(devices * (iend - istart)).start()
    p_sam = PBar(devices * sample_size).start()
    for n in range(devices):
        device = device_factory.wwp_s_37(n, n)
        if sim:
            # Simulate
            data = simulate(device, istart, iend, p_sim, newline=True)
            if export:
                P_el = resample(data['P_el'], 15)
                fn = '/tmp/hp%03d.csv' % n
                np.savetxt(fn, P_el / 1000.0, delimiter=',')
            else:
                plot_sim(drange(start, end, timedelta(minutes=1)), device, data)
        if sam:
            # Make sample
            sample = create_sample(device, sample_size, istart, iend, p_sam)
            if export:
                fn = '/tmp/hp%03d_samples.csv' % n
                np.savetxt(fn, sample / 1000.0, delimiter=',')
            else:
                plot_sample(drange(start, end, timedelta(minutes=15)), sample)
    print()
