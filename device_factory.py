from appliancesim.ext.device import Device, Converter, SuccessiveSampler
from appliancesim import data as appdata
from appliancesim.ext.thermal import HeatDemand
from chpsim.CHP import Storage, Scheduler
import chpsim.CHP as chp
import heatpumpsim as hp
from batterysim.battery import Battery, Scheduler as BatSched


def ecopower_1(seed, id):
    return create_heater(seed, id, model='Vaillant EcoPower 1.0',
        T_min=273+50, T_max=273+70, storage_weight=500, storage_loss=1.5,
        annual_demand=6000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def ecopower_3(seed, id):
    return create_heater(seed, id, model='Vaillant EcoPower 3.0',
        T_min=273+50, T_max=273+70, storage_weight=1000, storage_loss=1.5,
        annual_demand=14000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def ecopower_4(seed, id):
    return create_heater(seed, id, model='Vaillant EcoPower 4.7',
        T_min=273+50, T_max=273+70, storage_weight=1500, storage_loss=1.5,
        annual_demand=17000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def ecopower_20(seed, id):
    return create_heater(seed, id, model='Vaillant EcoPower 20.0',
        T_min=273+50, T_max=273+70, storage_weight=8000, storage_loss=1.5,
        annual_demand=50000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def se_wpf_5(seed, id):
    return create_heater(seed, id, model='Stiebel Eltron WPF 5',
        T_min=273+40, T_max=273+50, storage_weight=300, storage_loss=1.5,
        annual_demand=10000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def se_wpf_7(seed, id):
    return create_heater(seed, id, model='Stiebel Eltron WPF 7',
        T_min=273+40, T_max=273+50, storage_weight=400, storage_loss=1.5,
        annual_demand=15000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def se_wpf_10(seed, id):
    return create_heater(seed, id, model='Stiebel Eltron WPF 10',
        T_min=273+40, T_max=273+50, storage_weight=600, storage_loss=1.5,
        annual_demand=20000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def se_wpf_13(seed, id):
    return create_heater(seed, id, model='Stiebel Eltron WPF 13',
        T_min=273+40, T_max=273+50, storage_weight=700, storage_loss=1.5,
        annual_demand=25000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def wwp_s_24(seed, id):
    return create_heater(seed, id, model='Weishaupt WWP S 24',
        T_min=273+40, T_max=273+50, storage_weight=1500, storage_loss=1.5,
        annual_demand=50000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def wwp_s_30(seed, id):
    return create_heater(seed, id, model='Weishaupt WWP S 30',
        T_min=273+40, T_max=273+50, storage_weight=2000, storage_loss=1.5,
        annual_demand=70000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def wwp_s_37(seed, id):
    return create_heater(seed, id, model='Weishaupt WWP S 37',
        T_min=273+40, T_max=273+50, storage_weight=3000, storage_loss=1.5,
        annual_demand=90000, P_noise=0.1, S_noise=0.1, D_noise=0.2, T_noise=5)


def rf_100(seed, id):
    return create_battery(seed, id, model='RedoxFlow 100 kWh',
        E_max=100000, eff_bat=0.752, eff_conv=0.93, idle_discharge=150,
        P_min=-10000, P_max=10000)


CHP_MODELS = {
    'Vaillant EcoPower 1.0': ecopower_1,
    'Vaillant EcoPower 3.0': ecopower_3,
    'Vaillant EcoPower 4.7': ecopower_4,
    'Vaillant EcoPower 20.0': ecopower_20,
}

HP_MODELS = {
    'Stiebel Eltron WPF 5': se_wpf_5,
    'Stiebel Eltron WPF 7': se_wpf_7,
    'Stiebel Eltron WPF 10': se_wpf_10,
    'Stiebel Eltron WPF 13': se_wpf_13,
    'Weishaupt WWP S 24': wwp_s_24,
    'Weishaupt WWP S 30': wwp_s_30,
    'Weishaupt WWP S 37': wwp_s_37,
}

BATTERY_MODELS = {
    'RedoxFlow 100 kWh': rf_100,
}


def create_heater(seed, id, model, T_min, T_max, storage_weight, storage_loss,
                  annual_demand, P_noise, S_noise, D_noise, T_noise):
    if model in list(CHP_MODELS.keys()):
        # Erstelle BHKW
        if model == 'Vaillant EcoPower 1.0':
            device = Device('chp', id, [Converter(), HeatDemand(), Storage(),
                    chp.Engine(), Scheduler(), chp.BoostHeater(),
                    SuccessiveSampler()], seed=seed)
        else:
            device = Device('chp', id, [Converter(), HeatDemand(), Storage(),
                    chp.Engine(), Scheduler(), chp.BoostHeater(),
                    SuccessiveSampler()], seed=seed)
        # Minimale Verweilzeit je gefahrenem Betriebsmodus
        device.components.engine.min_step_duration = 60
        # initiale Werte für Verlaufs-Schätzer
        # device.components.storage.T_env = 18
        # device.components.engine.T_delta = 0
    elif model in list(HP_MODELS.keys()):
        # Erstelle Wärmepumpe
        # device = Device('heatpump', id, [Converter(), HeatDemand(),
        #         hp.RandomHeatSource(), Storage(), hp.Engine(), Scheduler(),
        #         chp.BoostHeater(), SuccessiveSampler()], seed=seed)
        device = Device('heatpump', id, [Converter(), HeatDemand(),
                hp.RandomHeatSource(), Storage(), hp.Engine(), Scheduler(),
                chp.BoostHeater(), SuccessiveSampler()], seed=seed)
    else:
        raise(TypeError('unknown model:', model))


    def noisy(tup):
        return tuple([int(device.random.normal(t, P_noise * t))
                      for t in tup[:-1]]) + (tup[-1],)

    def noisyl(lis):
        return [int(device.random.normal(t, P_noise * t)) for t in lis]

    # Leistungsdaten BHKW
    #   args: P_el_min, P_el_max, P_th_min, P_th_max, modulation steps
    engine = device.components.engine
    if model == 'Vaillant EcoPower 1.0':
        # no modulation, set steps directly
        engine.steps = [noisy((0, 0)), noisy((1000, 2500))]
    elif model == 'Vaillant EcoPower 3.0':
        # Drehzahl: 1400-3600 in 100er Schritten modulierbar --> 22 Stufen
        engine.set_equidistant_steps(*noisy((1500, 3000, 4700, 8000, 22)))
    elif model == 'Vaillant EcoPower 4.7':
        # Drehzahl: 1400-3600 in 100er Schritten modulierbar --> 22 Stufen
        engine.set_equidistant_steps(*noisy((1500, 4700, 4700, 12500, 22)))
    elif model == 'Vaillant EcoPower 20.0':
        # neu, noch kein Datenblatt verfügbar
        engine.set_equidistant_steps(*noisy((7000, 20000, 12000, 42000, 22)))
        # FIXME: Modulationsstufen

    # Leistungsdaten Wärmepumpe
    #   args: power curves (el and th) according to data sheet of the device
    elif model == 'Stiebel Eltron WPF 5':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([2000, 2400])},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [noisyl([5900, 5600]), noisyl([8900, 8400])],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 7':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([2500, 3100])},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [noisyl([6500, 6300]), noisyl([11900, 11300])],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 10':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([2400, 3200])},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [noisyl([8700, 8200]), noisyl([15800, 14900])],
                }
            }
        }
    elif model == 'Stiebel Eltron WPF 13':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([4200, 5200])},
                'P_th': {
                    'grid': [[-5, 20], [35, 50]],
                    'values': [noisyl([11600, 11200]), noisyl([21300, 20200])],
                }
            }
        }
    elif model == 'Weishaupt WWP S 24':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([5800, 7800])},
                'P_th': {
                    'grid': [[-5, 25], [35, 50]],
                    'values': [noisyl([21100, 20000]), noisyl([41500, 39000])],
                }
            }
        }
    elif model == 'Weishaupt WWP S 30':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {
                    'grid': [[35, 45, 55]],
                    'values': noisyl([7500, 9000, 11000])
                },
                'P_th': {
                    'grid': [[-5, 25], [35, 45, 55]],
                    'values': [noisyl([26500, 25500, 24500]),
                               noisyl([54500, 53000, 51000])],
                }
            }
        }
    elif model == 'Weishaupt WWP S 37':
        device.components.engine.characteristics = {
            'setpoints': {
                'P_el': {'grid': [[35, 50]], 'values': noisyl([8500, 11500])},
                'P_th': {
                    'grid': [[-5, 25], [35, 50]],
                    'values': [noisyl([32000, 29500]), noisyl([63500, 61500])],
                }
            }
        }

    else:
        raise(TypeError('unknown model:', model))

    # Hysterese-Korridor
    device.components.engine.T_min = T_min
    device.components.engine.T_max = T_max
    device.components.boost_heater.T_min = T_min
    device.components.boost_heater.P_el_nominal = storage_weight / 200  # kW
    # Warmwasserspeicher
    device.components.storage.weight = \
            device.random.normal(storage_weight, S_noise * storage_weight)
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
    device.components.heatsink.weather_file = \
            appdata.dwd_weather('bremen', 2010)
    # Wärme-Jahresverbrauch
    # EFH: >15000
    # ZFH: >25000
    # MFH: >45000
    # HH: >150000
    device.components.heatsink.annual_demand = \
            device.random.normal(annual_demand, D_noise * annual_demand)
    # Rauschen auf dem VDI-Wärmebedarf
    device.components.heatsink.temp_noise = T_noise

    # Berechnung des Wärmebedarfes (Erklärung von Ontje):
    # vdi 4655 hat basiszeitreihen für 10 typtage für elektrische, heiz und
    # warmwasserlast. für die zeitreihen sind je nach klimazone gewichte
    # angegeben. ich benutze die temperaturen und bedeckungsgrad aus den
    # wetterdaten um den richtigen typtag zu ermitteln, interpoliere dann noch
    # zwischen den temperaturen und pack noch ein rauschen drauf.

    return device


def create_battery(seed, id, model, E_max, eff_bat, eff_conv, idle_discharge,
                   P_min, P_max):

    if model in list(BATTERY_MODELS.keys()):
        device = Device('battery', id, [Converter(), Battery(),
                        BatSched(), SuccessiveSampler()], seed=seed)
    else:
        raise(TypeError('unknown model:', model))

    # Leistungsdaten
    device.components.battery.max_E = E_max
    device.components.battery.efficiency_battery = eff_bat
    device.components.battery.efficiency_converter = eff_conv
    device.components.battery.idle_discharge = idle_discharge
    device.components.battery.P_min = P_min
    device.components.battery.P_max = P_max

    return device
