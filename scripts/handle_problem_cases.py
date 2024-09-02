"""
Script for pre-processing problem cases,
where automatic identification of start and stop samples failed
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_DIR, PROBLEM_CASES

first_click = True
start = 0
end = 0

combined_dataset = 0

def onclick(event):
    global first_click
    global start
    global end
    global combined_dataset
    if first_click:
        start = int(event.xdata)
        first_click = False
    else:
        end = int(event.xdata)
        first_click = True

        truncated_data = combined_dataset[start:end, :]
        np.save(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}.npy', truncated_data)

fnames = [fname for fname in os.listdir(DATA_DIR) if fname.endswith('.txt')]

for idx, fname in enumerate(fnames):
    exp_number = int(os.path.splitext(fname)[0].split('_')[-1])
    # data = np.loadtxt(f'{DATA_DIR}/{fname}', skiprows=8, usecols=[1, 2, 3])
    # defl = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}_filtered.npz')

    # combined_dataset = np.c_[data, defl['dx'], defl['dy']]

    combined_dataset = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}.npy')

    if exp_number in PROBLEM_CASES:
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # for idx in range(5):
            # axs[idx].plot(combined_dataset[:, idx])
        axs.plot(combined_dataset[:, 1])
        plt.show()
        plt.close()
        fig.canvas.mpl_disconnect(cid)
    else:
        np.save(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}.npy', combined_dataset)


