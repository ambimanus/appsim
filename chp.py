import random

import numpy as np

from appliancesim.ext.device import Device, Consumer, SuccessiveSampler
from appliancesim import data as appdata
from appliancesim.ext.thermal import HeatDemand
from chpsim.CHP import Storage, Engine, Scheduler, StubBoostHeater


def create_device(seed, id):
    # Reproduzierbarkeit
    rng = random.Random()
    np.random.seed(seed)
    rng.seed(seed)

    # Erstelle BHKW
    device = Device('chp', id,
        [Consumer(), HeatDemand(), Storage(), Engine(), Scheduler(),
            StubBoostHeater(), SuccessiveSampler()], seed=seed)

    # irgendein Vaillant Modell
    device.components.engine.set_equidistant_steps(1300, 4700, 4000, 12500, 5)

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
    # initiale Werte für Verlaufs-SChätzer
    # device.components.storage.T_env = 18
    # device.components.engine.T_delta = 0
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