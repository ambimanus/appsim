import sys
import os
import datetime

import scenario_factory


now = datetime.datetime.now()
ts = now.isoformat().split('T')[0]

sc = scenario_factory.Scenario()
sc.from_JSON(sys.argv[1])
sc.rev_appsim = sys.argv[2]

datadir = 'data'
d = str(os.path.join(os.getcwd(), datadir, '_'.join((ts, sc.title))))
if not os.path.exists(d):
    os.makedirs(d)
fn = str(os.path.join(d, '.'.join((str(sc.seed), 'json'))))
if os.path.exists(fn):
    raise RuntimeError('File already exists: %s' % fn)
sc.save_JSON(fn)

print(fn)
