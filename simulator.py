import os
import xdrlib
import pickle
from tempfile import NamedTemporaryFile

import numpy as np

from progress import PBar


def resample(d, resolution):
    # resample the innermost axis to 'resolution'
    shape = tuple(d.shape[:-1]) + (int(d.shape[-1]/resolution), resolution)
    return d.reshape(shape).sum(-1)/resolution


def simulate(device, start, end, progress, newline=False):
    headers = ['P_el', 'P_th', 'T', 'T_env']
    data = np.zeros((len(headers), end - start))

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


def create_sample(device, sample_size, t_start, t_end, progress, density=None, noise=False):
    if density is None:
        if device.typename == 'heatpump':
            density = 0.03
        elif device.typename == 'chp':
            density = 0.01
        else:
            raise RuntimeError('unknown type: %s' % device.typename)
    device = device.copy()
    d = (t_end - t_start)
    if d == 0:
        return np.zeros((sample_size, 0)), np.zeros((sample_size, 0)), []
    states = None
    sim_data = None
    if hasattr(device.components, 'sampler'):
        sampler = device.components.sampler
        sampler.setpoint_density = density
        sample = np.array(sampler.sample(sample_size, duration=int(d / 15), perfect=1))
    elif hasattr(device.components, 'special_sampler'):
        sampler = device.components.special_sampler
        sampler.setpoint_density = density
        # Apply different sample approaches
        sim_data_1, states_1 = sampler.sample_old(sample_size / 4, duration=int(d / 15), perfect=0)
        sim_data_2, states_2 = sampler.sample_old(sample_size / 4, duration=int(d / 15), perfect=1)
        sim_data_3, states_3 = sampler.sample(sample_size / 4, duration=int(d / 15), perfect=0)
        sim_data_4, states_4 = sampler.sample(sample_size / 4, duration=int(d / 15), perfect=1)
        # Combine data
        sim_data = sim_data_1 + sim_data_2 + sim_data_3 + sim_data_4
        states = states_1 + states_2 + states_3 + states_4
        # Data reshaping
        sim_data = np.array(sim_data).swapaxes(0, 1)
        sample = np.array(sim_data[0])
        if device.typename == 'heatpump':
            # This is a consumer, so negate P_el
            sim_data[0] = sim_data[0] * (-1.0)
    elif hasattr(device.components, 'minmax_sampler'):
        sampler = device.components.minmax_sampler
        # sampler.setpoint_density = density    # not used in this sampler
        sample = np.array(sampler.sample(sample_size, duration=int(d / 15)))
    elif hasattr(device.components, 'modulating_sampler'):
        sampler = device.components.modulating_sampler
        # sampler.setpoint_density = density    # not used in this sampler
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
    return sample, states, sim_data


def run_unctrl(sc):
    print('--- Simulating uncontrolled behavior (full)')
    p_sim = PBar(len(sc.devices) * (sc.i_end - sc.i_pre)).start()
    sim_data = {}
    sc.state_files_unctrl = {}
    for d in sc.devices:
        aid = str(d.typename) + str(d.id)
        # Create a shadow copy of the device to operate on
        d = d.copy()
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, p_sim)
        # Simulate [i_start, block_start]
        sim_data_b_start = simulate(d, sc.i_start, sc.i_block_start, p_sim)
        # Simulate [block_start, block_end]
        sim_data_b_end = simulate(d, sc.i_block_start, sc.i_block_end, p_sim)
        # Save state (needed in run_states() below)
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files_unctrl[aid] = tmpf.name
        # Simulate [block_end, i_end]
        sim_data_i_end = simulate(d, sc.i_block_end, sc.i_end, p_sim)
        # Combine sim_data
        sim_data[aid] = np.concatenate(
                (sim_data_b_start, sim_data_b_end, sim_data_i_end), axis=1)
    print()
    return sim_data


def run_pre(sc):
    print('--- Simulating uncontrolled behavior in [pre, start - 1] and generating samples')
    progress = PBar((len(sc.devices) * (sc.i_start - sc.i_pre)) +
                    (len(sc.devices) * sc.sample_size * (sc.i_block_end -
                            sc.i_start))).start()
    sample_data = {}
    states_data = {}
    sample_sim_data = {}
    sc.state_files = {}
    if sc.i_block_end - sc.i_block_start == 0:
        return (np.zeros((len(sc.devices), 4, (sc.i_block_start - sc.i_start) / 15)),
                np.zeros((len(sc.devices), sc.sample_size, 0)),
                np.zeros((len(sc.devices), sc.sample_size, 0)))
    for d in sc.devices:
        aid = str(d.typename) + str(d.id)
        # Pre-Simulation
        simulate(d, sc.i_pre, sc.i_start, progress)
        # Simulation
        # sim_data[aid] = simulate(d, sc.i_start, sc.i_block_start, progress)
        # Save state
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files[aid] = tmpf.name
        # Sampling
        sample, states, s_sim_data = create_sample(d, sc.sample_size,
                sc.i_start, sc.i_block_end, progress, noise=sc.svsm)
        sample_data[aid] = sample
        states_data[aid] = states
        sample_sim_data[aid] = s_sim_data
    print()
    return sample_data, states_data, sample_sim_data


def run_schedule(sc):
    print('--- Simulating controlled behaviour in [block_start, block_end]')
    p_sim = PBar(len(sc.devices) * (sc.i_block_end - sc.i_block_start)).start()
    basedir = os.path.dirname(sc.loaded_from)
    schedules = np.load(os.path.join(basedir, sc.sched_file))
    sim_data = {}
    sc.state_files_ctrl = {}
    for d in sc.devices:
        aid = str(d.typename) + str(d.id)
        # Load state
        statefile = sc.state_files[aid]
        with open(statefile, 'rb') as data:
            unpacker = xdrlib.Unpacker(data.read())
            d.load_state(unpacker)
        os.remove(statefile)
        sched = schedules[aid]
        if d.typename == 'heatpump':
            # This is a consumer, so negate P_el
            sched = sched * (-1.0)
        # Set schedule
        d.components.scheduler.schedule = sched.tolist()
        # Simulate
        sim_data[aid] = simulate(d, sc.i_block_start, sc.i_block_end, p_sim)
        # Save state
        packer = xdrlib.Packer()
        d.save_state(packer)
        tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
        tmpf.write(packer.get_buffer())
        tmpf.close()
        sc.state_files_ctrl[aid] = tmpf.name
    print()
    return sim_data


# Alternative to run_schedule() which directly uses the states for the
# respective sample that was selected by COHDA, instead of simulating the
# schedule (states have been produced by the sampler for each generated sample,
# see create_sample())
def run_state(sc):
    print('--- Loading controlled behaviour for selected samples in [block_start, block_end]')
    assert hasattr(sc, 'run_pre_statesfile')
    basedir = os.path.dirname(sc.loaded_from)
    schedules = np.load(os.path.join(basedir, sc.sched_file))
    all_samples = np.load(os.path.join(basedir, sc.run_pre_samplesfile))
    sample_sim_data = np.load(os.path.join(basedir, sc.run_pre_samples_simdatafile))
    sim_data = {}
    with open(os.path.join(basedir, sc.run_pre_statesfile), 'rb') as infile:
        all_states = pickle.load(infile)
    sc.state_files_ctrl = {}
    for d in sc.devices:
        aid = str(d.typename) + str(d.id)
        sched = schedules[aid]
        samples = all_samples[aid]
        ssd = sample_sim_data[aid]
        # !for aidtmp in sc.aids: print('%s: %s' % (aidtmp, np.where((all_samples[aidtmp] == sched).all(axis=1))))
        try:
            # Search schedule in samples
            idx = np.where((samples == sched).all(axis=1))[0][0]
            state = all_states[aid][idx]
            sim_data[aid] = ssd.swapaxes(0, 1)[idx]
            tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
            tmpf.write(state)
            tmpf.close()
            sc.state_files_ctrl[aid] = tmpf.name
        except IndexError:
            # Schedule not found in sample, device keeps its initial (uncontrolled) schedule
            t_start, b_start, b_end = sc.t_start, sc.t_block_start, sc.t_block_end
            div = 1
            if (b_end - t_start).total_seconds() / 60 == sched.shape[-1] * 15:
                div = 15
            elif (b_end - t_start).total_seconds() / 60 == sched.shape[-1] * 60:
                div = 60
            # b_s = (b_start - sc.t_start).total_seconds() / 60 / div
            b_e = (b_end - sc.t_start).total_seconds() / 60 / div
            unctrl = np.load(os.path.join(basedir, sc.run_unctrl_datafile))
            sim_data[aid] = unctrl[aid][:,:b_e]
            sc.state_files_ctrl[aid] = sc.state_files_unctrl[aid]
    print()
    return sim_data


def run_post(sc):
    print('--- Simulating uncontrolled behavior in [block_end, end]')
    p_sim = PBar(len(sc.devices) * (sc.i_end - sc.i_block_end)).start()
    if sc.i_block_end - sc.i_block_start == 0:
        return np.zeros((len(sc.devices), 4, (sc.i_end - sc.i_block_end) / 15))
    sim_data = {}
    for d in sc.devices:
        aid = str(d.typename) + str(d.id)
        statefile = sc.state_files_ctrl[aid]
        # Load state
        with open(statefile, 'rb') as data:
            unpacker = xdrlib.Unpacker(data.read())
            d.load_state(unpacker)
        os.remove(statefile)
        # Simulate
        sim_data[aid] = simulate(d, sc.i_block_end, sc.i_end, p_sim)
    print()
    return sim_data
