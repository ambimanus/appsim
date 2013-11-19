import sys
import os
import shutil
import datetime

import scenario_factory


now = datetime.datetime.now()
ts = now.isoformat().split('T')[0]

sc = scenario_factory.Scenario()
sc.from_JSON(sys.argv[1])
sc.rev_appsim = sys.argv[2]

datadir = 'data'
d = str(os.path.join(os.getcwd(), datadir, '_'.join((ts, sc.title))))
if os.path.exists(d):
    print('"%s" already exists, may I delete it? [Y/n] ' % d, end='', file=sys.stderr)
    delete = input()
    if delete == 'Y' or delete == '':
        shutil.rmtree(d)
    else:
        sys.exit(1)
os.makedirs(d)
fn = str(os.path.join(d, '.'.join((str(sc.seed), 'json'))))
sc.save_JSON(fn)

print(fn)
