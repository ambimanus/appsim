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

        self._make_timestamps()
        self._make_devices()


if __name__ == '__main__':
    sc = Large(1)
    sc.save_JSON('/tmp/sc.json')

    sc1 = Scenario()
    sc1.load_JSON('/tmp/sc.json')

    import simulate
    sim_data, sample_data = simulate.run(sc1)
    print()
    print(len(sim_data), len(sample_data))
    for d in sim_data:
        print(d['P_el'].shape)
    for d in sample_data:
        print(d.shape)
