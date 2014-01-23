import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import epoch2num

import device_factory


if __name__ == '__main__':
    amount = 50
    devices = []
    for i in range(amount):
        device = device_factory.ecopower_4(i, i)

        devices.append(device)

    start = int(time.mktime(datetime(2010, 1, 2).timetuple()) // 60)
    end = int(time.mktime(datetime(2010, 1, 3).timetuple()) // 60)

    sample_time = start + 15 * 24
    sample_dur = 16

    P = [[] for d in devices]
    T = [[] for d in devices]
    Th = [[] for d in devices]
    for now in range(start, sample_time):
        for idx, device in enumerate(devices):
            device.step(now)
            P[idx].append(device.components.consumer.P)
            T[idx].append(device.components.storage.T)
            Th[idx].append(device.components.heatsink.in_heat)

    samples = []
    for d in devices:
        # d.components.sampler.setpoint_density = 0.1
        samples.append(d.components.sampler.sample(100, sample_dur))
    # samples = [d.components.sampler.sample(100, sample_dur) for d in devices]

    schedule = np.zeros(sample_dur)
    for idx, device in enumerate(devices):
        # min_schedule_idx = np.argmin(np.sum(np.abs(samples[idx]), axis=1))
        # device.components.scheduler.schedule = samples[idx][min_schedule_idx]
        # schedule += samples[idx][min_schedule_idx]

        max_schedule_idx = np.argmax(np.sum(np.abs(samples[idx]), axis=1))
        device.components.scheduler.schedule = samples[idx][max_schedule_idx]
        schedule += samples[idx][max_schedule_idx]

    for now in range(sample_time, end):
        for idx, device in enumerate(devices):
            device.step(now)
            P[idx].append(device.components.consumer.P)
            T[idx].append(device.components.storage.T)
            Th[idx].append(device.components.heatsink.in_heat)

    P = np.sum(P, axis=0)
    Th = np.sum(Th, axis=0)
    T = np.mean(T, axis=0)

    ax = plt.subplot(2, 1, 1)
    ax.grid(True)
    tz = 60  # timezone deviation in minutes
    x = epoch2num(np.arange((start + tz) * 60, (end + tz) * 60, 60))
    Th = np.reshape(Th, (len(x) // 15, 15)).mean(axis=1)
    ax.plot_date(x[::15], Th, color='magenta', label='P$_{th,out}$ (kW)', ls='-',
            marker=None)
    ax.legend()
    ax = plt.subplot(2, 1, 2, sharex=ax)
    ax.grid(True)
    l1 = ax.plot_date(x, P, label='P$_{el}$ (kW)', ls='-', marker=None)
    sched_x = epoch2num(np.arange(
            (sample_time + tz) * 60, ((sample_time + tz) + sample_dur * 15) * 60, 60))
    l2 = ax.plot_date(sched_x[::15], schedule, color='r', label='Schedule',
            ls='-', marker=None)
    ax = plt.twinx()
    l3 = ax.plot_date(x, T, color='g', label='T (\\textdegree C)', ls='-', marker=None)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    plt.gcf().autofmt_xdate()


    # # Samples plot
    # fig, ax = plt.subplots(len(samples))
    # if len(samples) == 1:
    #     ax = [ax]
    # for i, sample in enumerate(samples):
    #     t = np.arange(len(sample[0]))
    #     for s in sample:
    #         ax[i].plot(t, s)

    plt.show()
