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
sim_dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'pre', 'npy'))))
if os.path.exists(sim_dfn):
    raise RuntimeError('File already exists: %s' % sim_dfn)
sam_dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'samples', 'npy'))))
if os.path.exists(sam_dfn):
    raise RuntimeError('File already exists: %s' % sam_dfn)
mod_dfn = str(os.path.join(d, '.'.join((str(sc.seed), 'modes', 'npy'))))
if os.path.exists(mod_dfn):
    raise RuntimeError('File already exists: %s' % mod_dfn)

sim_data, modes_data, sample_data = simulator.run_pre(sc)
np.save(sim_dfn, sim_data)
np.save(sam_dfn, sample_data)
np.save(mod_dfn, modes_data)
sc.run_pre_datafile = os.path.basename(sim_dfn)
sc.run_pre_samplesfile = os.path.basename(sam_dfn)
sc.run_pre_modesfile = os.path.basename(mod_dfn)
sc.save_JSON(sc_file)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(len(sample_data))
# for i, samples in enumerate(sample_data):
#     t = np.arange(samples.shape[1])
#     for s in samples:
#         ax[i].plot(t, s)
# plt.show()
# sys.exit(1)
