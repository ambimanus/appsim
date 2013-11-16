import os
from mercurial import hg, ui, commands


def get_repo_root():
    path = os.path.dirname(os.path.realpath(__file__))
    while os.path.exists(path):
        if '.hg' in os.listdir(path):
            return path
        path = os.path.realpath(os.path.join(path, '..'))
    raise RuntimeError('No .hg repository found!')


ui = ui.ui()
repo_path = get_repo_root()
repo = hg.repository(ui, repo_path)
ui.pushbuffer()
commands.identify(ui, repo, rev='.')
print ui.popbuffer().split()[0]
