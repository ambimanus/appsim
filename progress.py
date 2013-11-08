import progressbar.progressbar as pbar

"""
pip install progressbar-latest
"""

class Progress(object):
    p = ['|', '/', '--', '\\']

    def __init__(self, divisor=10):
        self.counter = 0
        self.divisor = divisor

    def __str__(self):
        s = Progress.p[(self.counter // self.divisor) % len(Progress.p)]
        self.counter += 1
        return s


class GeneratorSpeed(pbar.widgets.AbstractWidget):
    def __init__(self):
        self.fmt = 'Speed: %d/s'
    def update(self, pbar):
        if pbar.seconds_elapsed < 2e-6:#== 0:
            bps = 0.0
        else:
            bps = float(pbar.currval) / pbar.seconds_elapsed
        return self.fmt % bps


class PBar(pbar.ProgressBar):
    def __init__(self, maxval):
        pbar.ProgressBar.__init__(self, widgets=[pbar.widgets.Percentage(), ' ',
                pbar.widgets.Bar(), ' ', pbar.widgets.ETA(), ' ', GeneratorSpeed()],
                maxval=maxval)

    def update(self, value=None):
        if value is None:
            pbar.ProgressBar.update(self, self.currval + 1)
        else:
            pbar.ProgressBar.update(self, value)