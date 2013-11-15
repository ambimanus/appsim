from appliancesim.ext.device import Device, Consumer, SuccessiveSampler
from appliancesim import data as appdata
from appliancesim.ext.thermal import HeatDemand
from chpsim.CHP import Storage, Scheduler
import chpsim.CHP as chp
import heatpumpsim as hp


def ecopower_1(seed, id):
    return create_device(seed, id, model='Vaillant EcoPower 1.0',
        T_min=273+50, T_max=273+70, storage_weight=500, storage_loss=1.5,
        annual_demand=10000, T_noise=2)


def ecopower_3(seed, id):
    return create_device(seed, id, model='Vaillant EcoPower 3.0',
        T_min=273+50, T_max=273+70, storage_weight=1000, storage_loss=1.5,
        annual_demand=20000, T_noise=2)


def ecopower_4(seed, id):
    return create_device(seed, id, model='Vaillant EcoPower 4.7',
        T_min=273+50, T_max=273+70, storage_weight=1500, storage_loss=1.5,
        annual_demand=35000, T_noise=2)


def ecopower_20(seed, id):
    return create_device(seed, id, model='Vaillant EcoPower 20.0',
        T_min=273+50, T_max=273+70, storage_weight=6000, storage_loss=1.5,
        annual_demand=140000, T_noise=2)


def se_wpf_5(seed, id):
    return create_device(seed, id, model='Stiebel Eltron WPF 5',
        T_min=273+40, T_max=273+50, storage_weight=300, storage_loss=1.5,
        annual_demand=10000, T_noise=2)


def se_wpf_7(seed, id):
    return create_device(seed, id, model='Stiebel Eltron WPF 7',
        T_min=273+40, T_max=273+50, storage_weight=400, storage_loss=1.5,
        annual_demand=15000, T_noise=2)


def se_wpf_10(seed, id):
    return create_device(seed, id, model='Stiebel Eltron WPF 10',
        T_min=273+40, T_max=273+50, storage_weight=600, storage_loss=1.5,
        annual_demand=20000, T_noise=2)


def se_wpf_13(seed, id):
    return create_device(seed, id, model='Stiebel Eltron WPF 13',
        T_min=273+40, T_max=273+50, storage_weight=700, storage_loss=1.5,
        annual_demand=25000, T_noise=2)


def wwp_s_24(seed, id):
    return create_device(seed, id, model='Weishaupt WWP S 24',
        T_min=273+40, T_max=273+50, storage_weight=1500, storage_loss=1.5,
        annual_demand=50000, T_noise=2)


def wwp_s_30(seed, id):
    return create_device(seed, id, model='Weishaupt WWP S 30',
        T_min=273+40, T_max=273+50, storage_weight=2000, storage_loss=1.5,
        annual_demand=70000, T_noise=2)


def wwp_s_37(seed, id):
    return create_device(seed, id, model='Weishaupt WWP S 37',
        T_min=273+40, T_max=273+50, storage_weight=3000, storage_loss=1.5,
        annual_demand=90000, T_noise=2)


CHP_MODELS = {'Vaillant EcoPower 1.0': ecopower_1,
              'Vaillant EcoPower 3.0': ecopower_3,
              'Vaillant EcoPower 4.7': ecopower_4,
              'Vaillant EcoPower 20.0': ecopower_20,
              }

HP_MODELS = {'Stiebel Eltron WPF 5': se_wpf_5,
             'Stiebel Eltron WPF 7': se_wpf_7,
             'Stiebel Eltron WPF 10': se_wpf_10,
             'Stiebel Eltron WPF 13': se_wpf_13,
             'Weishaupt WWP S 24': wwp_s_24,
             'Weishaupt WWP S 30': wwp_s_30,
             'Weishaupt WWP S 37': wwp_s_37,
             }


def create_device(seed, id, model, T_min, T_max, storage_weight, storage_loss,
                  annual_demand, T_noise):
    if model in list(CHP_MODELS.keys()):
        # Erstelle BHKW
        device = Device('chp', id, [Consumer(), HeatDemand(), Storage(),
                chp.Engine(), Scheduler(), chp.StubBoostHeater(),
                SuccessiveSampler()], seed=seed)
        # Minimale Verweilzeit je gefahrenem Betriebsmodus
        device.components.engine.min_step_duration = 60
        # initiale Werte für Verlaufs-Schätzer
        # device.components.storage.T_env = 18
        # device.components.engine.T_delta = 0
    elif model in list(HP_MODELS.keys()):
        # Erstelle Wärmepumpe
        device = Device('heatpump', id, [Consumer(), HeatDemand(),
                hp.RandomHeatSource(), Storage(), hp.Engine(), Scheduler(),
                SuccessiveSampler()], seed=seed)
    else:
        raise(TypeError('unknown model:', model))

    # Leistungsdaten BHKW
    #   args: P_el_min, P_el_max, P_th_min, P_th_max, modulation steps
    engine = device.components.engine
    if model == 'Vaillant EcoPower 1.0':
        # no modulation, set steps directly
        engine.steps = [(0, 0), (1000, 2500)]
    elif model == 'Vaillant EcoPower 3.0':
        # Drehzahl: 1400-3600 in 100er Schritten modulierbar --> 22 Stufen
        engine.set_equidistant_steps(1500, 3000, 4700, 8000, 22)
    elif model == 'Vaillant EcoPower 4.7':
        # Drehzahl: 1400-3600 in 100er Schritten modulierbar --> 22 Stufen
        engine.set_equidistant_steps(1500, 4700, 4700, 12500, 22)
    elif model == 'Vaillant EcoPower 20.0':
        # neu, noch kein Datenblatt verfügbar
        engine.set_equidistant_steps(7000, 20000, 12000, 42000, 22)
        # FIXME: Modulationsstufen

    # Leistungsdaten Wärmepumpe
    #   args: power curves (el and th) according to data sheet of the device
    elif model == 'Stiebel Eltron WPF 5':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [2000, 2400]},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [[5900, 5600], [8900, 8400]],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 7':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [2500, 3100]},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [[6500, 6300], [11900, 11300]],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 10':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [2400, 3200]},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [[8700, 8200], [15800, 14900]],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 13':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [4200, 5200]},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [[11600, 11200], [21300, 20200]],
                }
            }
        }
    elif model == 'Weishaupt WWP S 24':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [5800, 7800]},
                'P_th': {
                    'grid': [[-5, 25], [35, 50]],
                    'values': [[21100, 20000], [41500, 39000]],
                }
            }
        }
    elif model == 'Weishaupt WWP S 30':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {
                    'grid': [[35, 45, 55]],
                    'values': [7500, 9000, 11000]
                },
                'P_th': {
                    'grid': [[-5, 25], [35, 45, 55]],
                    'values': [[26500, 25500, 24500], [54500, 53000, 51000]],
                }
            }
        }
    elif model == 'Weishaupt WWP S 37':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': [8500, 11500]},
                'P_th': {
                    'grid': [[-5, 25], [35, 50]],
                    'values': [[32000, 29500], [63500, 61500]],
                }
            }
        }

    else:
        raise(TypeError('unknown model:', model))

    # Hysterese-Korridor
    device.components.engine.T_min = T_min
    device.components.engine.T_max = T_max
    # Warmwasserspeicher
    device.components.storage.weight = storage_weight
    # Bereitschaftsenergieverbrauch
    device.components.storage.loss = storage_loss
    # initiale Temperatur zufällig
    device.components.storage.T = device.random.uniform(T_min, T_max)
    # DWD Referenz-Witterungsverlauf für Region TRY03
    # (siehe appliancesim/ext/thermal/demand.pxi)
    device.components.heatsink.set_building('TRY03', 'efh')
    # VDI Referenzlastprofil für KWK in EFH/MFH
    device.components.heatsink.norm_consumption_file = appdata.vdi_4655()
    # DWD Wetterdaten für 2010
    device.components.heatsink.weather_file = appdata.dwd_weather('bremen', 2010)
    # Wärme-Jahresverbrauch
    # EFH: >15000
    # ZFH: >25000
    # MFH: >45000
    # HH: >150000
    device.components.heatsink.annual_demand = annual_demand
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = T_noise

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    return device
