import time
import random
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import drange

from appliancesim.ext.device import Device, Consumer, SuccessiveSampler
from appliancesim import data
from appliancesim.ext.thermal import HeatDemand
from heatpumpsim import Storage, Engine, RandomHeatSource

import progressbar.progressbar as pbar


class Progress(object):
    p = ['|', '/', '--', '\\']

    def __init__(self, divisor=10):
        self.counter = 0
        self.divisor = divisor

    def __str__(self):
        s = Progress.p[(self.counter // self.divisor) % len(Progress.p)]
        self.counter += 1
        return s


class GeneratorSpeed(pbar.widgets.AbstractWidget):
    def __init__(self):
        self.fmt = 'Speed: %d/s'
    def update(self, pbar):
        if pbar.seconds_elapsed < 2e-6:#== 0:
            bps = 0.0
        else:
            bps = float(pbar.currval) / pbar.seconds_elapsed
        return self.fmt % bps


class PBar(pbar.ProgressBar):
    def __init__(self, maxval):
        pbar.ProgressBar.__init__(self, widgets=[pbar.widgets.Percentage(), ' ',
                pbar.widgets.Bar(), ' ', pbar.widgets.ETA(), ' ', GeneratorSpeed()],
                maxval=maxval)

    def update(self, value=None):
        if value is None:
            pbar.ProgressBar.update(self, self.currval + 1)
        else:
            pbar.ProgressBar.update(self, value)


if __name__ == '__main__':
    # Reproduzierbarkeit
    seed = 0
    rng = random.Random()
    np.random.seed(0)
    rng.seed(0)

    # Erstelle Wärmepumpe
    device = Device('heatpump', 0, [Consumer(), HeatDemand(),
            RandomHeatSource(), Storage(), Engine(),
            SuccessiveSampler()], seed=seed)

    # Warmwasserspeicher: SBP 200 E
    device.components.storage.weight = 200
    # Bereitschaftsenergieverbrauch
    device.components.storage.loss = 1.5
    # initiale Temperatur zufällig 35-60 °C
    device.components.storage.T = 273 + rng.uniform(35, 60)
    # DWD Referenz-Witterungsverlauf für Region TRY03
    # (siehe appliancesim/ext/thermal/demand.pxi)
    device.components.heatsink.set_building('TRY03', 'efh')
    # VDI Referenzlastprofil für KWK in EFH/MFH
    device.components.heatsink.norm_consumption_file = data.vdi_4655()
    # DWD Wetterdaten für 2010
    device.components.heatsink.weather_file = data.dwd_weather('bremen', 2010)
    # Korrektur des Wärme-Jahresverbrauchs (Einheitenproblem --> Ontje FIXME)
    device.components.heatsink.annual_demand = 100000 * 60 * 10
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = 2

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    # Datenfelder
    P_el = []
    P_th = []
    storage_T = []
    in_heat = []

    # Simulationszeit
    start, end = datetime(2010, 1, 1), datetime(2010, 12, 31)
    delta = timedelta(minutes=1)
    t = drange(start, end, delta)

    # Simulation
    istart = int(time.mktime(start.timetuple()) // 60)
    iend = int(time.mktime(end.timetuple()) // 60)
    progress = PBar(iend - istart).start()
    for now in range(istart, iend):
        device.step(now)
        P_el.append(device.components.engine.P_el)
        P_th.append(device.components.engine.P_th)
        storage_T.append(device.components.storage.T)
        in_heat.append(device.components.heatsink.in_heat)
        progress.update(now - istart)

    print()
    print('heatsink.annual_demand = %.2f' % device.components.heatsink.annual_demand)
    print('heatsink.in_heat = %.2f' % (sum(in_heat)//60000))
    print('Differenz: %.2f' % (device.components.heatsink.annual_demand - (sum(in_heat)//60000)))

    # # Visualisierung
    # fig, ax = plt.subplots(2, sharex=True)

    # ax[0].plot_date(t, storage_T, fmt='-', lw=1, label='storage T')
    # # ax[0].plot_date(t, in_heat, fmt='-', lw=1, label='storage inheat')
    # leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
    #                     borderaxespad=0.0, fancybox=False)

    # ax[1].plot_date(t, P_el, fmt='-', lw=1, label='P$_{el}$')
    # ax[1].plot_date(t, P_th, fmt='-', lw=1, label='P$_{th}$')
    # leg1 = ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
    #                     borderaxespad=0.0, fancybox=False)

    # for label in leg0.get_texts() + leg1.get_texts():
    #     label.set_fontsize('x-small')
    # fig.autofmt_xdate()
    # fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    # plt.show()
