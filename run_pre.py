import sys
import os
import datetime

import numpy as np

import scenario_factory
import simulator


sc_file = sys.argv[1]
sc = scenario_factory.Scenario()
sc.load_JSON(sc_file)
sc.run_pre_ts = datetime.datetime.now()

d = os.path.dirname(sc_file)
dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'pre', 'npy'))))
if os.path.exists(dfn):
    raise RuntimeError('File already exists: %s' % dfn)

data = simulator.run_pre(sc)
np.save(dfn, data)
sc.run_pre_datafile = dfn
sc.save_JSON(sc_file)
