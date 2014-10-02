import sys
import os

import scenario_factory


sc_file = sys.argv[1]
sc = scenario_factory.Scenario()
sc.load_JSON(sc_file)
d = os.path.dirname(sc_file)

delete = []
for f in ['run_pre_statesfile', 'state_files', 'state_files_ctrl', 'state_files_unctrl']:
    if hasattr(sc, f):
        fspec = getattr(sc, f)
        if isinstance(fspec, str):
            delete.append(os.path.join(d, fspec))
        elif isinstance(fspec, list):
            delete.extend([os.path.join(d, ffspec) for ffspec in fspec])
        elif isinstance(fspec, dict):
            delete.extend([os.path.join(d, ffspec) for ffspec in fspec.values()])
        delattr(sc, f)
for f in delete:
    try:
        os.remove(f)
    except:
        pass

sc.save_JSON(sc_file)
