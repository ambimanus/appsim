import progressbar.progressbar as pbar

"""
pip install progressbar-latest
"""

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

    def flush(self):
        self.fd.write(self._format_line() + '\r')
