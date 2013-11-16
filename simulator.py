import os
import xdrlib
from tempfile import NamedTemporaryFile

import numpy as np

from progress import PBar


def simulate(device, start, end, progress, newline=False):
    # Datenfelder
    headers = ['P_el', 'P_th', 'T', 'T_env']
    data = np.empty((len(headers), end - start))

    # Simulation
    for now in range(start, end):
        device.step(now)
        i = now - start
        data[0][i] = device.components.engine.P_el
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


def create_sample(device, sample_size, t_start, t_end, progress, density=0.1):
    device = device.copy()
    device.step(t_start)
    device.components.sampler.setpoint_density = density
    d = (t_end - t_start) / 15
    sample = np.array(device.components.sampler.sample(sample_size, duration=d))
    # Add noise to prevent breaking the SVDD model due to linear dependencies
    np.random.seed(device.random.rand_int())
    noise = np.abs(np.random.normal(scale=0.0001, size=(sample_size, d)))

    sample = (sample / 1000) + noise

    if device.typename == 'heatpump':
        # This is a consumer, so negate the sample
        sample = sample * (-1.0)

    progress.update(progress.currval + sample_size)
    return sample


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
    p_sim = PBar(len(sc.devices) * (sc.i_block_start - sc.i_pre)).start()
    p_sam = PBar(len(sc.devices) * sc.sample_size).start()

    sim_data = []
    sample_data = []
    for d in sc.devices:
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, p_sim)
        # Simulation
        sim_data.append(simulate(d, sc.i_start, sc.i_block_start, p_sim))
        # Save state
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files.append(tmpf.name)
        # Sampling
        sample_data.append(create_sample(d, sc.sample_size, sc.i_block_start,
                                         sc.i_block_end, p_sam))
    print()
    return np.array(sim_data), sample_data


def run_schedule(sc):
    print('--- Simulating controlled behaviour in [block_start, block_end]')
    p_sim = PBar(len(sc.devices) * (sc.i_block_end - sc.i_block_start)).start()
    schedules = np.load(sc.sched_file)
    sim_data = []
    for d, statefile, sched in zip(sc.devices, sc.state_files, schedules):
        # Load state
        with open(statefile, 'rb') as data:
            unpacker = xdrlib.Unpacker(data.read())
            d.load_state(unpacker)
        os.remove(statefile)
        # Set schedule
        d.components.scheduler.schedule = sched.tolist()
        # Simulate
        sim_data.append(simulate(d, sc.i_block_start, sc.i_block_end, p_sim))
    print()
    return np.array(sim_data)


def run_post(sc):
    print('--- Simulating uncontrolled behavior in [block_end, end]')
    p_sim = PBar(len(sc.devices) * (sc.i_end - sc.i_block_end)).start()
    sim_data = []
    for d in sc.devices:
        sim_data.append(simulate(d, sc.i_block_end, sc.i_end, p_sim))
    print()
    return np.array(sim_data)
