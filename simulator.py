import os
import xdrlib
from tempfile import NamedTemporaryFile

import numpy as np

from appliancesim.ext.device import SuccessiveSampler, HiResSampler

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
    if device.typename == 'battery':
        for now in range(start, end):
            device.step(now)
            i = now - start
            data[0][i] = device.components.consumer.P
            data[1][i] = None
            data[2][i] = device.components.battery.E / 1000
            data[3][i] = None
            progress.update()
    else:
        for now in range(start, end):
            device.step(now)
            i = now - start
            data[0][i] = device.components.engine.P_el
            if hasattr(device.components, 'boost_heater'):
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

    return resample(data, 15)


def create_sample(d, sample_size, t_start, t_end, progress, density=None):
    if density is None:
        if d.typename == 'heatpump':
            density = 0.25
        elif d.typename == 'chp':
            density = 0.75
        else:
            raise RuntimeError('unknown type: %s' % d.typename)
    device = d.copy()
    device.components.sampler.setpoint_density = density
    d = (t_end - t_start)
    if d == 0:
        return np.zeros((sample_size, 0)), np.zeros((sample_size, 0))
    sampler = device.components.sampler
    if type(sampler) == SuccessiveSampler:
        modes = None
        sample = np.array(sampler.sample(sample_size, duration=d / 15))
    elif type(sampler) == HiResSampler:
        modes, sample = np.array(sampler.sample(sample_size, duration=d)
                                ).swapaxes(0, 1)
        sample = resample(sample, 15)
    else:
        raise RuntimeError('unknown sampler %s' %type(sampler))
    # Add noise to prevent breaking the SVDD model due to linear dependencies
    np.random.seed(device.random.rand_int())
    scale = 0.000001 * np.max(np.abs(sample))
    noise = np.random.normal(scale=scale, size=sample.shape)

    sample = sample + noise

    if device.typename == 'heatpump':
        # This is a consumer, so negate the sample
        sample = sample * (-1.0)

    progress.update(progress.currval + (sample_size * d))
    return modes, sample


def run_unctrl(sc):
    print('--- Simulating uncontrolled behavior (full)')
    p_sim = PBar(len(sc.devices) * (sc.i_end - sc.i_pre)).start()
    sim_data = []
    for d in sc.devices:
        # Create a shadow copy of the device to operate on
        d = d.copy()
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, p_sim)
        # Simulation
        sim_data.append(simulate(d, sc.i_start, sc.i_end, p_sim))
    print()
    return np.array(sim_data)


def run_pre(sc):
    print('--- Simulating uncontrolled behavior in [pre, start - 1] and [start, block_start]')
    progress = PBar((len(sc.devices) * (sc.i_block_start - sc.i_pre)) +
                    (len(sc.devices) * sc.sample_size * (sc.i_block_end -
                            sc.i_block_start))).start()
    sim_data = []
    modes_data = []
    sample_data = []
    if sc.i_block_end - sc.i_block_start == 0:
        return (np.zeros((len(sc.devices), 4, (sc.i_block_start - sc.i_start) / 15)),
                np.zeros((len(sc.devices), sc.sample_size, 0)),
                np.zeros((len(sc.devices), sc.sample_size, 0)))
    for d in sc.devices:
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, progress)
        # Simulation
        sim_data.append(simulate(d, sc.i_start, sc.i_block_start, progress))
        # Save state
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files.append(tmpf.name)
        # Sampling
        modes, sample = create_sample(d, sc.sample_size, sc.i_block_start,
                                      sc.i_block_end, progress)
        modes_data.append(modes)
        sample_data.append(sample)
    print()
    return np.array(sim_data), np.array(modes_data), np.array(sample_data)


def run_schedule(sc):
    print('--- Simulating controlled behaviour in [block_start, block_end]')
    p_sim = PBar(len(sc.devices) * (sc.i_block_end - sc.i_block_start)).start()
    basedir = os.path.dirname(sc.loaded_from)
    schedules = np.load(os.path.join(basedir, sc.sched_file))
    samples_file = np.load(os.path.join(basedir, sc.run_pre_samplesfile))
    modes_file = np.load(os.path.join(basedir, sc.run_pre_modesfile))
    sim_data = []
    for d, statefile, sched, modes, samples in zip(
            sc.devices, sc.state_files, schedules, modes_file, samples_file):
        # Load state
        with open(statefile, 'rb') as data:
            unpacker = xdrlib.Unpacker(data.read())
            d.load_state(unpacker)
        os.remove(statefile)
        if hasattr(d.components, 'direct_scheduler'):
            # Find modes for schedule
            if sched in samples:
                # Matching sample found, select mode directly
                mode = modes[np.where(samples == sched)]
            else:
                # No matching sample, select the most similar one
                mode, dist = None, None
                for i, sample in enumerate(samples):
                    r = np.sqrt(np.sum((np.array(sched) - np.array(sample))**2))
                    if dist is None or r < dist:
                        dist = r
                        mode = modes[i]
            # Set operational mode
            d.components.direct_scheduler.schedule = mode.tolist()
        else:
            if d.typename == 'heatpump':
                # This is a consumer, so negate P_el
                sched = sched * (-1.0)
            # No modes available, use power schedule
            d.components.scheduler.schedule = sched.tolist()
        # Simulate
        sim_data.append(simulate(d, sc.i_block_start, sc.i_block_end, p_sim))
        # Save state
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files_ctrl.append(tmpf.name)
    print()
    return np.array(sim_data)


def run_post(sc):
    print('--- Simulating uncontrolled behavior in [block_end, end]')
    p_sim = PBar(len(sc.devices) * (sc.i_end - sc.i_block_end)).start()
    if sc.i_block_end - sc.i_block_start == 0:
        return np.zeros((len(sc.devices), 4, (sc.i_end - sc.i_block_end) / 15))
    sim_data = []
    for d, statefile in zip(sc.devices, sc.state_files_ctrl):
        # Load state
        with open(statefile, 'rb') as data:
            unpacker = xdrlib.Unpacker(data.read())
            d.load_state(unpacker)
        os.remove(statefile)
        # Simulate
        sim_data.append(simulate(d, sc.i_block_end, sc.i_end, p_sim))
    print()
    return np.array(sim_data)
