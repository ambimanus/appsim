import sys
import os
import datetime

import numpy as np

import scenario_factory
from progress import PBar


def resample(d, resolution):
    # resample the innermost axis to 'resolution'
    shape = tuple(d.shape[:-1]) + (int(d.shape[-1]/resolution), resolution)
    return d.reshape(shape).sum(-1)/resolution


def simulate(device, start, end, progress, newline=False):
    # Datenfelder
    headers = ['P_el', 'P_th', 'T', 'T_env']
    data = np.zeros((len(headers), end - start))

    # Simulation
    # if device.typename == 'battery':
    #     for now in range(start, end):
    #         device.step(now)
    #         i = now - start
    #         data[0][i] = device.components.consumer.P
    #         data[1][i] = None
    #         data[2][i] = device.components.battery.E / 1000
    #         data[3][i] = None
    #         progress.update()
    # else:
    for now in range(start, end):
        device.step(now)
        i = now - start
        data[0][i] = device.components.converter.P
        if hasattr(device.components, 'boost_heater'):
            if device.typename == 'chp':
                # the boost heater is a consumer, so substract its power
                data[0][i] -= device.components.boost_heater.P_el
            elif device.typename == 'hp':
                # the power array will be negated below, so just add the power
                data[0][i] += device.components.boost_heater.P_el
        data[1][i] = device.components.engine.P_th
        data[2][i] = device.components.storage.T
        data[3][i] = device.components.heatsink.T_env
        progress.update()

    if device.typename == 'heatpump':
        # This is a consumer, so negate P_el
        data[0] = data[0] * (-1.0)

    progress.flush()
    if newline:
        print()

    return data


def create_sample(d, sample_size, t_start, t_end, progress, density=None, noise=False):
    if density is None:
        if d.typename == 'heatpump':
            density = 0.03
        elif d.typename == 'chp':
            density = 0.01
        else:
            raise RuntimeError('unknown type: %s' % d.typename)
    device = d.copy()
    d = (t_end - t_start)
    if d == 0 or sample_size == 0:
        return np.zeros((sample_size, 0)), np.zeros((sample_size, 0))
    if hasattr(device.components, 'sampler'):
        sampler = device.components.sampler
        sampler.setpoint_density = density
        sample = np.array(sampler.sample(sample_size, duration=int(d / 15)))
    else:
        raise RuntimeError('unknown sampler %s' %type(sampler))

    if noise:
        # Add noise to prevent breaking the SVDD model due to linear dependencies
        np.random.seed(device.random.rand_int())
        scale = 0.000001 * np.max(np.abs(sample))
        noise = np.random.normal(scale=scale, size=sample.shape)
        sample = sample + noise

    if device.typename == 'heatpump':
        # This is a consumer, so negate the sample
        sample = sample * (-1.0)

    progress.update(progress.currval + (sample_size * d))
    return sample


def run(sc):
    progress = PBar(
            # Pre-Simulation
            (len(sc.devices) * (sc.i_start - sc.i_pre)) +
            # Simulation
            (len(sc.devices) * (sc.i_end - sc.i_start)) +
            # Samples
            (len(sc.devices) * sc.sample_size * (sc.i_end - sc.i_start))
    ).start()
    sim_data = []
    sample_data = []
    # print('--- Simulating init phase [pre, start - 1]')
    for d in sc.devices:
        simulate(d, sc.i_pre, sc.i_start, progress)
    # print('--- Simulating [start, end]')
    for d in sc.devices:
        sim_data.append(simulate(d, sc.i_start, sc.i_end, progress))
    # print('--- Generating %d samples for [start, end]' % sc.sample_size)
    for d in sc.devices:
        sample = create_sample(d, sc.sample_size, sc.i_start, sc.i_end,
                                      progress, noise=sc.sample_noise)
        sample_data.append(sample)
    print()
    return np.array(sim_data), np.array(sample_data)


if __name__ == '__main__':
    sc_file = sys.argv[1]
    sc = scenario_factory.Scenario()
    sc.load_JSON(sc_file)
    sc.run_timestamp = datetime.datetime.now()

    d = os.path.dirname(sc_file)
    sim_dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'simulation', 'npy'))))
    if os.path.exists(sim_dfn):
        raise RuntimeError('File already exists: %s' % sim_dfn)
    sam_dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'samples', 'npy'))))
    if os.path.exists(sam_dfn):
        raise RuntimeError('File already exists: %s' % sam_dfn)

    sim_data, sample_data = run(sc)
    np.save(sim_dfn, sim_data)
    np.save(sam_dfn, sample_data)
    sc.simulation_file = os.path.basename(sim_dfn)
    sc.samples_file = os.path.basename(sam_dfn)
    sc.save_JSON(sc_file)
