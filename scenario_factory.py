import random
import time
import json
from datetime import datetime

from device_factory import CHP_MODELS, HP_MODELS, BATTERY_MODELS


class Scenario(object):

    def __init__(self):
        self.loaded_from = None
        self.stored_to = None

    def _make_devices(self):
        self.devices = []
        idx = 0
        for name, n in self.device_templates:
            if name in CHP_MODELS:
                func = CHP_MODELS[name]
            elif name in HP_MODELS:
                func = HP_MODELS[name]
            elif name in BATTERY_MODELS:
                func = BATTERY_MODELS[name]
            else:
                raise TypeError('unknown type: %s' % name)
            for i in range(n):
                self.devices.append(func(idx, idx))
                idx += 1


    def _make_timestamps(self):
        self.i_pre, self.i_start, self.i_end = [
                int(time.mktime(t.timetuple()) // 60) for t in (
                        self.t_pre, self.t_start, self.t_end)]


    def from_JSON(self, js):
        if type(js) == str:
            js = json.loads(js)
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


    def load_JSON(self, filename):
        # Read file
        with open(filename, 'r') as fp:
            js = json.load(fp)

        self.from_JSON(js)
        self.loaded_from = filename


    def to_JSON(self):
        d = dict(self.__dict__)
        for k in list(d.keys()):
            if k == 'devices' or k == 'rnd' or k[:2] == 'i_':
                del d[k]

        return json.dumps(d, indent=2, cls=ScenarioEncoder)


    def save_JSON(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.to_JSON())
        self.stored_to = filename


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
        self.title = 'Test'
        self.seed = seed
        self.sample_size = 10
        self.sample_noise = True

        self.t_pre = datetime(2010, 3, 25)
        self.t_start = datetime(2010, 4, 1)
        self.t_end = datetime(2010, 4, 4)

        self.device_templates = [
            ('Vaillant EcoPower 1.0', 1),
            ('Stiebel Eltron WPF 10', 1),
        ]

        self.rnd = random.Random(seed)
        self._make_timestamps()
        self._make_devices()


def resample(d, resolution):
    # resample the innermost axis to 'resolution'
    shape = tuple(d.shape[:-1]) + (int(d.shape[-1]/resolution), resolution)
    return d.reshape(shape).sum(-1)/resolution


if __name__ == '__main__':
    # # Reproduzierbarkeit
    # np.random.seed(seed)

    sc = Test(0)

    # Test JSON export/import:
    # sc.save_JSON('/tmp/sc.json')
    # sc1 = Scenario()
    # sc1.load_JSON('/tmp/sc.json')
    # sc, sc1 = Test(0), Test(0)
    # print(sc)
    # import sys
    # sys.exit()

    import simulator
    sim_data, samples = simulator.run(sc)
    resolution = 1
    if resolution > 1:
        # resample data to "resolution" minutes, e.g. 15
        sim_data = simulator.resample(sim_data, resolution)

    # Plot
    from datetime import timedelta
    from matplotlib import pyplot as plt
    from matplotlib.dates import drange

    t = drange(sc.t_start, sc.t_end, timedelta(minutes=resolution))

    for device in sim_data:
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].set_ylabel('P$_{el}$ [kW]')
        ymax = device[0].max() / 1000.0
        ax[0].set_ylim(-0.01, ymax + (ymax * 0.1))
        ax[0].plot_date(t, device[0] / 1000.0, fmt='-', lw=1)

        ax[1].set_ylabel('T$_{storage}$ [\\textdegree C]')
        ax[1].plot_date(t, device[2] - 273.0, fmt='-', lw=1)

        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)

    plt.show()
