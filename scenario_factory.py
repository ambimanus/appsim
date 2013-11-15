import random
import time
import json
from datetime import datetime

from device_factory import CHP_MODELS, HP_MODELS


class Scenario(object):

    def _make_devices(self):
        self.devices = []
        idx = 0
        for name, n in self.device_templates:
            if name in CHP_MODELS:
                func = CHP_MODELS[name]
            else:
                func = HP_MODELS[name]
            for i in range(n):
                self.devices.append(func(self.rnd.random(), idx))
                idx += 1


    def _make_timestamps(self):
        self.i_pre, self.i_start, self.i_block_start, self.i_block_end,\
            self.i_end = [int(time.mktime(t.timetuple()) // 60) for t in (
                          self.t_pre, self.t_start, self.t_block_start,
                          self.t_block_end, self.t_end)]


    def load_JSON(self, filename):
        # Read file
        with open(filename, 'r') as fp:
            js = json.load(fp)

        # Import data
        self.__dict__ = js

        # Recreate sub-instances
        for k in list(js.keys()):
            if k[:2] == 't_':
                tt = js[k]
                setattr(self, k, datetime(*tt[:6]))
        self.rnd = random.Random(self.seed)
        self._make_timestamps()
        self._make_devices()


    def to_JSON(self):
        d = self.__dict__
        if 'devices' in d:
            del d['devices']
        if 'rnd' in d:
            del d['rnd']

        return json.dumps(d, indent=2, cls=ScenarioEncoder)


    def save_JSON(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.to_JSON())


    def __str__(self):
        return self.to_JSON()


class ScenarioEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.timetuple()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class Test(Scenario):

    def __init__(self, seed):
        self.t_pre = datetime(2010, 4, 1)
        self.t_start = datetime(2010, 4, 2)
        self.t_block_start = datetime(2010, 4, 2, 12)
        self.t_block_end = datetime(2010, 4, 2, 13)
        self.t_end = datetime(2010, 4, 3)

        self.sample_size = 100
        self.seed = seed
        self.rnd = random.Random(seed)
        self.device_templates = [
            ('Vaillant EcoPower 1.0', 1),
            ('Vaillant EcoPower 3.0', 1),
            # ('Stiebel Eltron WPF 10', 1),     # save_state not yet implemented
        ]

        self.state_files = []
        self.sched_file = None

        self._make_timestamps()
        self._make_devices()


class Large(Scenario):

    def __init__(self, seed):
        self.t_pre = datetime(2010, 4, 1)
        self.t_start = datetime(2010, 4, 2)
        self.t_block_start = datetime(2010, 4, 2, 11)
        self.t_block_end = datetime(2010, 4, 2, 14)
        self.t_end = datetime(2010, 4, 3)

        self.sample_size = 100
        self.seed = seed
        self.rnd = random.Random(seed)
        self.device_templates = [
            ('Vaillant EcoPower 1.0', 10),
            ('Vaillant EcoPower 3.0', 5),
            ('Vaillant EcoPower 4.7', 5),
            ('Vaillant EcoPower 20.0', 1),
            ('Stiebel Eltron WPF 5', 5),
            ('Stiebel Eltron WPF 7', 5),
            ('Stiebel Eltron WPF 10', 5),
            ('Stiebel Eltron WPF 13', 5),
            ('Weishaupt WWP S 24', 1),
            ('Weishaupt WWP S 30', 1),
            ('Weishaupt WWP S 37', 1),
        ]

        self.state_files = []
        self.sched_file = None

        self._make_timestamps()
        self._make_devices()


if __name__ == '__main__':
    sc = Test(0)
    sc.save_JSON('/tmp/sc.json')

    sc1 = Scenario()
    sc1.load_JSON('/tmp/sc.json')

    def stats(data, samples=None):
        print(data.shape)
        if samples is not None:
            assert len(data) == len(samples)
            for s in samples:
                print(s.shape)

    # Data
    import numpy as np
    m, q = len(sc1.devices), sc1.i_end - sc1.i_start
    ctrl = np.empty((m, 4, q))

    import simulator
    # Uncontrolled: full
    unctrl = simulator.run_unctrl(sc1)
    # Uncontrolled: pre
    sim_data_pre, sample_data = simulator.run_pre(sc1)
    ctrl[:,:,:sc1.i_block_start - sc1.i_start] = sim_data_pre
    # Controlled
    from tempfile import NamedTemporaryFile
    sched = np.zeros((len(sc1.devices), (sc1.i_block_end - sc1.i_block_start) // 15))
    tmpf = NamedTemporaryFile(mode='wb', dir='/tmp', delete=False)
    np.save(tmpf, sched)
    tmpf.close()
    sc1.sched_file = tmpf.name
    sim_data_sched = simulator.run_schedule(sc1)
    ctrl[:,:,sc1.i_block_start - sc1.i_start:sc1.i_block_end - sc1.i_start] = sim_data_sched
    # Uncontrolled: post
    ctrl[:,:,sc1.i_block_end - sc1.i_start:sc1.i_end - sc1.i_start] = simulator.run_post(sc1)


    # Plot
    from datetime import timedelta
    from matplotlib import pyplot as plt
    from matplotlib.dates import drange

    t = drange(sc1.t_start, sc1.t_end, timedelta(minutes=1))

    for d_unctrl, d_ctrl in zip(unctrl, ctrl):
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_ylabel('P$_{el}$ [kW]')
        ax[0].set_ylim(-0.01, 2.0)
        ax[0].plot_date(t, d_unctrl[0] / 1000.0, fmt='-', lw=1, label='unctrl')
        ax[0].plot_date(t, d_ctrl[0] / 1000.0, fmt='-', lw=1, label='ctrl')
        leg0 = ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                            borderaxespad=0.0, fancybox=False)

        ax[1].set_ylabel('T$_{storage}$ [\\textdegree C]')
        ax[1].plot_date(t, d_unctrl[2] - 273.0, fmt='-', lw=1, label='unctrl')
        ax[1].plot_date(t, d_ctrl[2] - 273.0, fmt='-', lw=1, label='ctrl')
        leg1 = ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
                            borderaxespad=0.0, fancybox=False)

        fig.autofmt_xdate()
        for label in leg0.get_texts() + leg1.get_texts():
            label.set_fontsize('x-small')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    # ax[1].plot_date(t, data['T'] - 273, fmt='-', lw=1, label='T [\\textdegree C]')
    # ax[1].plot_date(t, data['T_env'], fmt='-', lw=1, label='T$_{env}$')
    # T_min = np.array([device.components.engine.T_min for x in t])
    # T_max = np.array([device.components.engine.T_max for x in t])
    # ax[1].plot_date(t, T_min - 273, fmt='k-', lw=1, label='T$_{min}$')
    # ax[1].plot_date(t, T_max - 273, fmt='k-', lw=1, label='T$_{max}$')
    # leg1 = ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4,
    #                     borderaxespad=0.0, fancybox=False)

    plt.show()


    # def plot_sample(t, sample):
    #     fig, ax = plt.subplots()
    #     ax.set_ylabel('P$_{el}$ [kW]')

    #     for s in sample:
    #         ax.plot_date(t, s, fmt='-', lw=1)

    #     fig.autofmt_xdate()
    #     fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.2, hspace=0.4)

    #     plt.show()


    # def resample(d, resolution):
    #     return (d.reshape(d.shape[0]/resolution, resolution).sum(1)/resolution)


