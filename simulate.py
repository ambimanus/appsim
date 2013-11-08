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

from progress import PBar


def simulate(seed, start, end, progress):
    # Reproduzierbarkeit
    rng = random.Random()
    np.random.seed(0)
    rng.seed(0)

    # Erstelle Wärmepumpe
    device = Device('heatpump', 0, [Consumer(), HeatDemand(),
            RandomHeatSource(), Storage(), Engine(),
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
    device.components.storage.weight = 200
    # Bereitschaftsenergieverbrauch
    device.components.storage.loss = 1.5
    # initiale Temperatur zufällig 35-50 °C
    device.components.storage.T = 273 + rng.uniform(35, 50)
    # DWD Referenz-Witterungsverlauf für Region TRY03
    # (siehe appliancesim/ext/thermal/demand.pxi)
    device.components.heatsink.set_building('TRY03', 'efh')
    # VDI Referenzlastprofil für KWK in EFH/MFH
    device.components.heatsink.norm_consumption_file = appdata.vdi_4655()
    # DWD Wetterdaten für 2010
    device.components.heatsink.weather_file = appdata.dwd_weather('bremen', 2010)
    # Korrektur des Wärme-Jahresverbrauchs (Einheitenproblem --> Ontje FIXME)
    device.components.heatsink.annual_demand = 60000
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = 2
    device.components.engine.T_min = 273 + 35
    device.components.engine.T_max = 273 + 45

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    # Datenfelder
    data = np.empty((4, (end - start)))

    # Simulation
    for now in range(start, end):
        device.step(now)
        i = now - start
        data[0][i] = device.components.engine.P_el
        data[1][i] = device.components.engine.P_th
        data[2][i] = device.components.storage.T
        data[3][i] = device.components.heatsink.in_heat
        progress.update()

    progress.flush()
    print()
    print('heatsink.annual_demand = %.2f' % device.components.heatsink.annual_demand)
    print('heatsink.in_heat = %.2f' % (sum(data[3])/60.0))

    return data


def resample(d, resolution):
    return (d.reshape(d.shape[0]/resolution, resolution).sum(1)/resolution)


if __name__ == '__main__':
    # Simulationszeit
    start, end = datetime(2010, 4, 1), datetime(2010, 4, 2)
    delta = timedelta(minutes=1)
    t = drange(start, end, delta)
    istart = int(time.mktime(start.timetuple()) // 60)
    iend = int(time.mktime(end.timetuple()) // 60)

    runs = 1
    progress = PBar(runs * (iend - istart)).start()
    for n in range(runs):
        data = simulate(0, istart, iend, progress)
        # Export
        P_el = resample(data[0], 15)
        fn = '/tmp/hp%3d.csv' % n
        np.savetxt(fn, P_el, delimiter=',')

    # Visualisierung
    fig, ax = plt.subplots(2, sharex=True)
    P_el, P_th, storage_t, in_heat = data

    ax[0].plot_date(t, storage_t - 273.0, fmt='-', lw=1, label='storage T')
    # ax[0].plot_date(t, in_heat, fmt='-', lw=1, label='storage inheat')
    leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    ax[1].plot_date(t, P_el, fmt='-', lw=1, label='P$_{el}$')
    ax[1].plot_date(t, P_th, fmt='-', lw=1, label='P$_{th}$')
    leg1 = ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                        borderaxespad=0.0, fancybox=False)

    for label in leg0.get_texts() + leg1.get_texts():
        label.set_fontsize('x-small')
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    plt.show()
