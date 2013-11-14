import random

from device_factory import CHP_MODELS, HP_MODELS


class Large(object):

    def __init__(self, seed):
        self.b_start = 11
        self.b_end = 14
        self.rnd = random.Random(seed)
        self.device_templates = [
            (CHP_MODELS['Vaillant EcoPower 1.0'], 10),
            (CHP_MODELS['Vaillant EcoPower 3.0'], 5),
            (CHP_MODELS['Vaillant EcoPower 4.7'], 5),
            (CHP_MODELS['Vaillant EcoPower 20.0'], 1),
            (HP_MODELS['Stiebel Eltron WPF 5'], 5),
            (HP_MODELS['Stiebel Eltron WPF 7'], 5),
            (HP_MODELS['Stiebel Eltron WPF 10'], 5),
            (HP_MODELS['Stiebel Eltron WPF 13'], 5),
            (HP_MODELS['Weishaupt WWP S 24'], 1),
            (HP_MODELS['Weishaupt WWP S 30'], 1),
            (HP_MODELS['Weishaupt WWP S 37'], 1),
        ]

        self.devices = []
        idx = 0
        for func, n in self.device_templates:
            for i in range(n):
                self.devices.append(func(self.rnd.random(), idx))
                idx += 1


s = Large(1)
