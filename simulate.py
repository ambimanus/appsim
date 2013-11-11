import time
import random
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import drange

from appliancesim.ext.device import Device, Consumer, SuccessiveSampler
from appliancesim import data as appdata
from appliancesim.ext.thermal import HeatDemand
from heatpumpsim import Storage, Engine, RandomHeatSource
from chpsim.CHP import Scheduler

from progress import PBar


def create_device(seed):
    # Reproduzierbarkeit
    rng = random.Random()
    np.random.seed(0)
    rng.seed(0)

    # Erstelle Wärmepumpe
    device = Device('heatpump', 0, [Consumer(), HeatDemand(),
            RandomHeatSource(), Storage(), Engine(), Scheduler(),
            SuccessiveSampler()], seed=seed)

    # Stiebel Eltron WPF 10
    device.components.engine.characteristics = {
        'setpoints': {
            'P_el': {'grid': [[35, 50]], 'values': [2.4, 3.2]},
            'P_th': {
                'grid': [[-5, 20], [35, 50]],
                'values': [[8.7, 8.2], [15.8, 14.9]],
            }
        }
    }
    # from heatpumpsim import Interpolator
    # from mpl_toolkits.mplot3d import Axes3D
    # interp = Interpolator(*device.components.engine.P_th_characteristic)
    # ax = plt.subplot(1, 1, 1, projection='3d')
    # ax.scatter(*interp.support_grid())
    # Z = []
    # X, Y = np.linspace(0, 50, 20), np.linspace(0, 50, 20)
    # for y in Y:
    #     Z.append([])
    #     for x in X:
    #         Z[-1].append(interp(x, y))
    # X, Y = np.meshgrid(X, Y)
    # ax.plot_surface(X, Y, Z)
    # plt.show()

    # Warmwasserspeicher: SBP 200 E
    device.components.storage.weight = 1000
    # Bereitschaftsenergieverbrauch
    device.components.storage.loss = 1.5
    # initiale Temperatur zufällig 35-50 °C
    device.components.storage.T = 273 + rng.uniform(35, 45)
    # DWD Referenz-Witterungsverlauf für Region TRY03
    # (siehe appliancesim/ext/thermal/demand.pxi)
    device.components.heatsink.set_building('TRY03', 'efh')
    # VDI Referenzlastprofil für KWK in EFH/MFH
    device.components.heatsink.norm_consumption_file = appdata.vdi_4655()
    # DWD Wetterdaten für 2010
    device.components.heatsink.weather_file = appdata.dwd_weather('bremen', 2010)
    # a = np.load(device.components.heatsink.norm_consumption_file)
    # import pdb
    # pdb.set_trace()
    # Korrektur des Wärme-Jahresverbrauchs (Einheitenproblem --> Ontje FIXME)
    device.components.heatsink.annual_demand = 40000 / 60
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = 2
    # Hysterese-Korridor
    device.components.engine.T_min = 273 + 35
    device.components.engine.T_max = 273 + 45

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    return device


def simulate(device, start, end, progress):
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
    print()
    # print('heatsink.annual_demand = %.2f' % device.components.heatsink.annual_demand)
    # print('heatsink.in_heat = %.2f' % (sum(data['in_heat'])/60.0))

    return data


def create_sample(device, sample_size, t_start, t_end, density=0.1):
    assert (t_end - t_start) / 15 == 96, '%d (only 96-dimensional samples supported)' % (t_end - t_start)
    device = device.copy()
    device.step(t_start)
    device.components.sampler.setpoint_density = density
    return device.components.sampler.sample(sample_size)


def plot_sim(t, device, data):
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot_date(t, data['P_el'], fmt='-', lw=1, label='P$_{el}$')
    ax[0].plot_date(t, data['P_th'], fmt='-', lw=1, label='P$_{th}$')
    leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    ax[1].plot_date(t, data['T'] - 273, fmt='-', lw=1, label='T')
    ax[1].plot_date(t, data['T_env'] - 273, fmt='-', lw=1, label='T$_{env}$')
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


if __name__ == '__main__':
    # Simulationszeit
    start, end = datetime(2010, 4, 1), datetime(2010, 4, 2)
    istart = int(time.mktime(start.timetuple()) // 60)
    iend = int(time.mktime(end.timetuple()) // 60)

    runs = 1
    progress = PBar(runs * (iend - istart)).start()
    for n in range(runs):
        # Create device
        device = create_device(n)
        # Simulate
        data = simulate(device, istart, iend, progress)
        # Make sample
        sample = create_sample(device, 100, istart, iend)
        t = drange(start, end, timedelta(minutes=15))
        plot_sample(t, sample)
        # # Export
        # P_el = resample(data['P_el'], 15)
        # fn = '/tmp/hp%3d.csv' % n
        # np.savetxt(fn, P_el, delimiter=',')
        # # Visualize
        # t = drange(start, end, timedelta(minutes=1))
        # plot_sim(t, device, data)

