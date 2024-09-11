"""Data processing"""

import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

import misc

import pywt
from scipy import signal
from matplotlib import pyplot as plt
plt.rc('axes', axisbelow=True)
plt.rc('font', family='Arial')
from matplotlib.patches import Rectangle
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plot_utils import modify_axis, hist
from colors import dark2

from joblib import dump, load
from pathlib import Path

from config import (
    DATA_DIR,
    RANDOM_SEED,
    PARAM_FILE,
    FONTSIZE,
    PLOT_DIR,
    RESULTS_DIR,
    VERBOSE,
    PROCESSED_DIR,
    TEST_SIZE,
    INPUT_SIZE,
    PROBLEM_CASES
)

class DataProcessing:
    """Data processing class"""
    def __init__(self):
        np.random.seed(RANDOM_SEED)
        scenarios = self.read_raw()

        self.train, self.test = train_test_split(scenarios, test_size=TEST_SIZE)

        np.random.shuffle(self.train)

        self.train_concat = self.train[0]
        for scenario in self.train[1:]:
            self.train_concat = np.concatenate((self.train_concat, scenario))

        self.scaler = MinMaxScaler()

        self.train_concat[:, :INPUT_SIZE] = self.scaler.fit_transform(self.train_concat[:, :INPUT_SIZE])

        for test_idx, test_scenario in enumerate(self.test):
            scaled_test_scenario = self.scaler.transform(test_scenario[:, :INPUT_SIZE])
            self.test[test_idx][:, :INPUT_SIZE] = scaled_test_scenario

    def get_train_test(self):
        return self.train_concat, self.test

    def get_scaler(self):
        return self.scaler

    def read_raw(self):
        """Method for processing raw data"""
        if any(Path(PROCESSED_DIR).iterdir()):
            scenarios = np.load(f'{PROCESSED_DIR}/processed_data.npy', allow_pickle=True)
        else:
            f_s = 100000
            n_edges = 2
            nb_scales = 100
            freq_div = 20
            wavelet = 'mexh'
            downsample_rate = 5
            sync_idx = 1
            sync_thresh = 0.07
            f_s_downsamples = f_s / downsample_rate

            params = pd.read_excel(PARAM_FILE)

            fnames = [fname for fname in os.listdir(DATA_DIR) if fname.endswith('.txt')]
            fnames.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

            scenarios = []

            for idx, fname in enumerate(tqdm(fnames)):
                # data = np.loadtxt(f'{DATA_DIR}/{fname}', skiprows=8, usecols=[1, 2, 3])
                # defl = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}_filtered.npz')
                # dx = defl['dx']
                # dy = defl['dy']

                exp_number = int(os.path.splitext(fname)[0].split('_')[-1])

                data = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}.npy')

                spsp, fz, ae, input_ae, r1, r2 = params[
                    params['Messdatei']==f'V0_{exp_number}'
                ].to_numpy()[0][1:7]

                chatter = params[params['Messdatei']==f'V0_{exp_number}'].to_numpy()[0][16]

                if chatter in ['+', 'o/+', 'o']:

                    data = signal.decimate(data, downsample_rate, axis=0)
                    # dx = signal.decimate(dx, downsample_rate)
                    # dy = signal.decimate(dy, downsample_rate)
                    time = np.array([1 / f_s_downsamples * t_idx for t_idx in range(len(data))])

                    #######################################################################################
                    ############################### Wavelet shizzle #######################################
                    #######################################################################################

                    # Use wavelets for high frequency resolution
                    fz_freq = spsp / 60.0 * n_edges
                    freqs = np.array([
                        fz_freq / freq_div + idx * fz_freq / freq_div for idx in range(nb_scales)
                    ])
                    scale = pywt.frequency2scale(wavelet, freqs / f_s_downsamples)

                    cwtmatr, __ = pywt.cwt(
                        (data[:, sync_idx] - np.mean(data[:, sync_idx])) / np.std(data[:, sync_idx]),
                        scale,
                        wavelet,
                        sampling_period=1/f_s_downsamples
                    )

                    power = np.power(np.abs(cwtmatr), 2)

                    # Scale-dependent normalization of power
                    scale_avg = np.repeat(scale, len(data)).reshape(power.shape)
                    power = power / scale_avg

                    start_idx = next(__ for __, val in enumerate(power[freq_div*2-1]) if val > sync_thresh)
                    sig_start = time[start_idx]
                    stop_idx = (
                        len(data) \
                        - next(__ for __, val in enumerate(np.flip(power[freq_div*2-1])) if val > sync_thresh)
                    )
                    sig_stop = time[stop_idx]

                    #######################################################################################
                    ################################# Signal plot #########################################
                    #######################################################################################

                    # if VERBOSE and exp_number in PROBLEM_CASES:
                    if VERBOSE:
                        print(f'Exp. number: {exp_number}')
                        force_labels = ['fx', 'fy', 'fz']
                        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

                        for jdx in range(3):
                            axs[0].plot(time, data[:, jdx], label=force_labels[jdx])

                        axs[0].axvline(sig_start)
                        axs[0].axvline(sig_stop)
                        axs[1].plot(time, data[:, 3] * 1e+6) # to µm
                        axs[1].plot(time, data[:, 4] * 1e+6) # to µm

                        axs[2].contourf(time, freqs, power, cmap='inferno')
                        axs[2].axhline(fz_freq*2)

                        axs[3].plot(time, power[freq_div*2-1])

                        fig.canvas.draw()

                        plt.setp(axs[0].get_xticklabels(), visible=False)
                        plt.setp(axs[1].get_xticklabels(), visible=False)
                        plt.setp(axs[2].get_xticklabels(), visible=False)

                        axs[0].set_ylabel('Force in N')
                        axs[1].set_ylabel('Deflection in µm')
                        axs[2].set_ylabel('Frequency in Hz')
                        axs[3].set_ylabel('Wavelet intensity')
                        axs[3].set_xlabel('Time in s')

                        fig.align_ylabels()
                        fig.tight_layout(pad=0.1)

                        plt.show()
                        plt.close()

                    #######################################################################################

                    data = data[start_idx:stop_idx]
                    # dx = dx[start_idx:stop_idx]
                    # dy = dy[start_idx:stop_idx]

                    time = np.array([1 / f_s_downsamples * t_idx for t_idx in range(len(data))])
                    features_target = np.c_[
                        time,
                        data[:, :3],
                        [spsp for __ in range(len(data))],
                        [fz for __ in range(len(data))],
                        [r1 for __ in range(len(data))],
                        [r2 for __ in range(len(data))],
                        data[:, 3:]
                        # dx,
                        # dy
                    ]

                    scenarios.append(features_target)

            scenarios = np.array(scenarios, dtype=object)

            np.save(f'{PROCESSED_DIR}/processed_data.npy', scenarios)

        return scenarios
