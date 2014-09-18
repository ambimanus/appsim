import sys
import os
import shutil
import datetime

import scenario_factory


def get_repo_root():
    path = os.path.dirname(os.path.realpath(__file__))
    while os.path.exists(path):
        if '.hg' in os.listdir(path):
            return path
        path = os.path.realpath(os.path.join(path, '..'))
    raise RuntimeError('No .hg repository found!')


def get_repo_revision():
    from mercurial import hg, ui, commands

    ui = ui.ui()
    repo_path = get_repo_root()
    repo = hg.repository(ui, repo_path)
    ui.pushbuffer()
    commands.identify(ui, repo, rev='.')
    return ui.popbuffer().split()[0]



now = datetime.datetime.now()
ts = now.isoformat().split('T')[0]

sc = scenario_factory.Scenario()
sc.from_JSON(sys.argv[1])
# sc.version = get_repo_revision()

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
fn = str(os.path.join(d, '.'.join((sc.title, 'json'))))
sc.save_JSON(fn)

print(fn)
