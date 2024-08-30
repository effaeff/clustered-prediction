"""Script for engagement detection using the wavelet transform"""

import os
import numpy as np
import pywt
import pandas as pd

from scipy import signal
from matplotlib import pyplot as plt

def detect_engagement():
    """Method for processing raw data"""
    f_s = 100000
    n_edges = 2
    nb_scales = 100
    freq_div = 20
    wavelet = 'mexh'
    downsample_rate = 5
    sync_idx = 1
    sync_thresh = 0.05
    f_s_downsamples = f_s / downsample_rate

    data_dir = 'data/01_raw'
    filtered_defl_dir = 'data/240408_Komp_Runout_ICME24/_eval'
    param_file = 'data/clustersim_lhs_Zuordnung_Messdaten.xlsx'

    params = pd.read_excel(param_file)

    fnames = [fname for fname in os.listdir(data_dir) if fname.endswith('.txt')]
    fnames.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    for idx, fname in enumerate(fnames):
        data = np.loadtxt(f'{data_dir}/{fname}', skiprows=8, usecols=[1, 2, 3, 4, 5, 6, 7])
        exp_number = os.path.splitext(fname)[0].split('_')[-1]
        print(f'Exp: {exp_number}')

        defl = np.load(f'{filtered_defl_dir}/Cluster_Sim_V0_{exp_number}_filtered.npz')
        dx = defl['dx']
        dy = defl['dy']

        spsp, fz, ae, input_ae, r1, r2 = params[
            params['Messdatei']==f'V0_{exp_number}'
        ].to_numpy()[0][1:7]


        data = signal.decimate(data, downsample_rate, axis=0)
        dx = signal.decimate(dx, downsample_rate)
        dy = signal.decimate(dy, downsample_rate)
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

        #######################################################################################
        ################################# Signal plot #########################################
        #######################################################################################

        force_labels = ['fx', 'fy', 'fz']
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        for jdx in range(3):
            axs[0].plot(time, data[:, jdx], label=force_labels[jdx])

        start_idx = next(__ for __, val in enumerate(power[freq_div*2-1]) if val > sync_thresh)
        sig_start = time[start_idx]
        stop_idx = (
            len(data) \
            - next(__ for __, val in enumerate(np.flip(power[freq_div*2-1])) if val > sync_thresh)
        )
        sig_stop = time[stop_idx]

        axs[0].axvline(sig_start)
        axs[0].axvline(sig_stop)
        axs[1].plot(time, dx * 1e+6) # to µm
        axs[1].plot(time, dy * 1e+6) # to µm

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

if __name__ == '__main__':
    detect_engagement()

