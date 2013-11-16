import sys
import os
import datetime

import numpy as np

import scenario_factory
import simulator

now = datetime.datetime.now()
ts = now.isoformat().split('T')[0]

sc = scenario_factory.Scenario()
sc.from_JSON(sys.argv[1])
sc.rev_appsim = sys.argv[2]
sc.ts_run_unctrl = now

basepath = 'data'
d = str(os.path.join(basepath, '_'.join((ts, sc.title))))
if not os.path.exists(d):
    os.makedirs(d)

sfn = str(os.path.join(d, '.'.join((str(sc.seed), 'json'))))
if os.path.exists(sfn):
    raise RuntimeError('File already exists: %s' % sfn)
sc.save_JSON(sfn)

dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'unctrl', 'npy'))))
if os.path.exists(dfn):
    raise RuntimeError('File already exists: %s' % dfn)

unctrl = simulator.run_unctrl(sc)
np.save(dfn, unctrl)

print(unctrl)
