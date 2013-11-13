import random

import numpy as np

from appliancesim.ext.device import Device, Consumer, SuccessiveSampler
from appliancesim import data as appdata
from appliancesim.ext.thermal import HeatDemand
from heatpumpsim import Engine, RandomHeatSource
from chpsim.CHP import Storage, Scheduler


def create_device(seed, id):
    # Reproduzierbarkeit
    rng = random.Random()
    np.random.seed(seed)
    rng.seed(seed)

    # Erstelle Wärmepumpe
    device = Device('heatpump', id, [Consumer(), HeatDemand(),
            RandomHeatSource(), Storage(), Engine(), Scheduler(),
            SuccessiveSampler()], seed=seed)

    # Stiebel Eltron WPF 10
    device.components.engine.characteristics = {
        'setpoints': {
            'P_el': {'grid': [[35, 50]], 'values': [2400, 3200]},
            'P_th': {
                'grid': [[-5, 20], [35, 50]],
                'values': [[8700, 8200], [15800, 14900]],
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

    # Hysterese-Korridor
    T_min, T_max = 273 + 50, 273 + 70
    device.components.engine.T_min = T_min
    device.components.engine.T_max = T_max
    # Warmwasserspeicher: SBP 200 E
    device.components.storage.weight = 500
    # Bereitschaftsenergieverbrauch
    device.components.storage.loss = 1.5
    # initiale Temperatur zufällig
    device.components.storage.T = rng.uniform(T_min, T_max)
    # DWD Referenz-Witterungsverlauf für Region TRY03
    # (siehe appliancesim/ext/thermal/demand.pxi)
    device.components.heatsink.set_building('TRY03', 'efh')
    # VDI Referenzlastprofil für KWK in EFH/MFH
    device.components.heatsink.norm_consumption_file = appdata.vdi_4655()
    # DWD Wetterdaten für 2010
    device.components.heatsink.weather_file = appdata.dwd_weather('bremen', 2010)
    # Wärme-Jahresverbrauch
    device.components.heatsink.annual_demand = 40000
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = 2

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    return device