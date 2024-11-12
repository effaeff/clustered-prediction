"""Data processing"""

import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

import misc

from geometry import ar_area

import pywt
from scipy import signal
from matplotlib import pyplot as plt
plt.rc('axes', axisbelow=True)
# plt.rc('font', family='Arial')
from matplotlib.patches import Rectangle
from matplotlib import colormaps as cm
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plot_utils import modify_axis, hist, CM_INCH
from series import butter_lowpass_filter
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
    PROBLEM_CASES,
    N_EDGES
)

class SelectionStandardizer:
    def __init__(self, data):
        self.click_cnt = 0
        self.data = data

        self.range_indices = np.zeros(6, dtype=int)

        self.cut_begin = 0
        self.cut_end = len(self.data)

        self.fig, self.axs = plt.subplots(figsize=(20, 10))
        self.axs.plot(data)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def get_data(self):
        return self.data

    def get_start_idx(self):
        return int(self.cut_begin)

    def get_end_idx(self):
        return int(self.cut_end)

    def onclick(self, event):
        self.range_indices[self.click_cnt] = int(event.xdata)
        if self.click_cnt < 5:
            self.axs.axvline(x=event.xdata, color='red')
            self.fig.canvas.draw()
            self.click_cnt += 1

            if self.click_cnt == 2:
                self.cut_begin = event.xdata
            elif self.click_cnt == 5:
                self.cut_end = event.xdata
        else:
            mean_start = np.mean(self.data[self.range_indices[0]:self.range_indices[1]])
            mean_middle = np.mean(self.data[self.range_indices[2]:self.range_indices[3]])
            mean_end = np.mean(self.data[self.range_indices[4]:self.range_indices[5]])

            idx_start = self.range_indices[0] + (self.range_indices[1] - self.range_indices[0])//2
            idx_middle = self.range_indices[2] + (self.range_indices[3] - self.range_indices[2])//2
            idx_end = self.range_indices[4] + (self.range_indices[5] - self.range_indices[4])//2

            first_line = np.linspace(mean_start, mean_middle, idx_middle - idx_start)
            second_line = np.linspace(mean_middle, mean_end, idx_end - idx_middle)

            self.axs.plot(range(idx_start, idx_middle), first_line, color='green')
            self.axs.plot(range(idx_middle, idx_end), second_line, color='green')
            self.axs.axvline(idx_middle, color='orange')

            self.data[idx_start:idx_middle] -= first_line
            self.data[idx_middle:idx_end] -= second_line

            # cut_sections = np.concatenate(
                # (self.data[self.range_indices[0]:self.range_indices[1]],
                # self.data[self.range_indices[2]:self.range_indices[3]],
                # self.data[self.range_indices[4]:self.range_indices[5]])
            # )

            # self.data -= np.mean(cut_sections)
            if plt.waitforbuttonpress():
                plt.close(self.fig)

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

        # Scaling moved to main script for now
        # self.scaler = MinMaxScaler()

        # self.train_concat[:, :INPUT_SIZE] = self.scaler.fit_transform(self.train_concat[:, :INPUT_SIZE])

        # for test_idx, test_scenario in enumerate(self.test):
            # scaled_test_scenario = self.scaler.transform(test_scenario[:, :INPUT_SIZE])
            # self.test[test_idx][:, :INPUT_SIZE] = scaled_test_scenario

    def get_train_test(self):
        return self.train_concat, self.test

    # def get_scaler(self):
        # return self.scaler

    def read_raw(self):
        """Method for processing raw data"""
        if os.path.isfile(f'{PROCESSED_DIR}/processed_data.npy'):
            scenarios = np.load(f'{PROCESSED_DIR}/processed_data.npy', allow_pickle=True)
        else:
            f_s = 100000
            n_edges = 2
            nb_scales = 100
            freq_div = 20
            wavelet = 'mexh'
            downsample_rate = 5
            sync_idx = 1
            sync_thresh = 0.05
            f_s_downsamples = f_s / downsample_rate

            params = pd.read_excel(PARAM_FILE)

            fnames = [fname for fname in os.listdir(DATA_DIR) if fname.endswith('.txt')]
            fnames.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

            scenarios = []

            for idx, fname in enumerate(tqdm(fnames)):
                # data = np.loadtxt(f'{DATA_DIR}/{fname}', skiprows=8, usecols=[1, 2, 3, 5, 6])
                # defl = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}_filtered.npz')
                # dx = defl['dx']
                # dy = defl['dy']
                exp_number = int(os.path.splitext(fname)[0].split('_')[-1])

                spsp, fz, ae, input_ae, r1, r2 = params[
                    params['Messdatei']==f'V0_{exp_number}'
                ].to_numpy()[0][1:7]
                chatter = params[params['Messdatei']==f'V0_{exp_number}'].to_numpy()[0][16]

                # if chatter in ['+', 'o/+', 'o']:
                if True:
                    if os.path.isfile(f'{PROCESSED_DIR}/{exp_number}_processed.npy'):
                        features_target = np.load(f'{PROCESSED_DIR}/{exp_number}_processed.npy')
                    else:
                        data = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}_combined.npy')
                        # data = np.load(f'{DATA_DIR}/Cluster_Sim_V0_{exp_number}.npy')

                        #######################################################################################
                        ############################ Deflection filtering #####################################
                        #######################################################################################

                        data[:, 3] = butter_lowpass_filter(data[:, 3], 10000, f_s, 6)
                        data[:, 4] = butter_lowpass_filter(data[:, 4], 10000, f_s, 6)

                        data = data[2500:]

                        before_std_dx = np.copy(data[:, 3])
                        before_std_dy = np.copy(data[:, 4])

                        selection_std_dx = SelectionStandardizer(data[:, 3])
                        dx_std = selection_std_dx.get_data()
                        selection_std_dy = SelectionStandardizer(data[:, 4])
                        dy_std = selection_std_dy.get_data()

                        __, ax_std = plt.subplots(2, 1)
                        ax_std[0].plot(
                            before_std_dx[selection_std_dx.get_start_idx():selection_std_dx.get_end_idx()],
                            label='dx'
                        )
                        ax_std[1].plot(dx_std[selection_std_dx.get_start_idx():selection_std_dx.get_end_idx()])
                        ax_std[0].plot(
                            before_std_dy[selection_std_dx.get_start_idx():selection_std_dx.get_end_idx()],
                            label='dy'
                        )
                        ax_std[1].plot(dy_std[selection_std_dx.get_start_idx():selection_std_dx.get_end_idx()])
                        ax_std[0].legend()
                        plt.show()

                        data[:, 3] = dx_std
                        data[:, 4] = dy_std

                        data = data[selection_std_dx.get_start_idx():selection_std_dx.get_end_idx()]

                        #######################################################################################

                        data = signal.decimate(data, downsample_rate, axis=0)
                        # dx = signal.decimate(dx, downsample_rate)
                        # dy = signal.decimate(dy, downsample_rate)
                        time = np.array([1 / f_s_downsamples * t_idx for t_idx in range(len(data))])

                        #######################################################################################
                        ################################## Path data ##########################################
                        #######################################################################################

                        path = np.genfromtxt(
                            f'{DATA_DIR}/V0_{exp_number}_XYZ_of_Pts.dat',
                            skip_header=6,
                            skip_footer=1,
                            delimiter=',',
                            usecols=[0, 1]
                        )

                        feed = fz * spsp * n_edges

                        s_per_mm = 60 / feed
                        path_dist = [0] + [
                            np.sqrt((path[jdx, 0] - path[jdx-1, 0])**2 + (path[jdx, 1] - path[jdx-1, 1])**2)
                            for jdx in range(1, len(path))
                        ]

                        # path_time = []
                        # for jdx, dist in enumerate(path_dist):
                            # local_time = s_per_mm * dist
                            # if jdx > 0:
                                # local_time += path_time[jdx-1]
                            # path_time.append(local_time)

                        # path_time = np.array(path_time)

                        # path_time_up = [0]
                        # jdx = 1
                        # while(path_time_up[-1] < path_time[-1]):
                            # path_time_up.append(1 / f_s_downsamples * jdx)
                            # jdx += 1

                        # upsampled_path_x = np.interp(path_time_up, path_time, path[:, 0])
                        # upsampled_path_y = np.interp(path_time_up, path_time, path[:, 1])

                        # path = np.transpose([upsampled_path_x, upsampled_path_y])

                        # path_cur_y = path[0, 1]
                        # while ar_area(4, path[0, 0], path_cur_y, -34.5, -34.5, 34.5, 34.5) > 0:
                            # path_cur_y -= (1 / s_per_mm) * 1 / f_s_downsamples
                            # path = np.insert(path, 0, [path[0, 0], path_cur_y], axis=0)
                            # path_time_up.append(path_time_up[-1] + 1 / f_s_downsamples)

                        # path_time_up = np.array(path_time_up)

                        overlap = np.array(
                            [ar_area(4, pos[0], pos[1], -34.5, -34.5, 34.5, 34.5) for pos in path]
                        )
                        overlap[overlap < 0] = 0
                        path_cutoff_end = (
                            len(overlap) \
                            - next(__ for __, val in enumerate(np.flip(overlap)) if val > 0)
                        )
                        overlap = overlap[:path_cutoff_end]
                        path = path[:path_cutoff_end]
                        # path_time_up = path_time_up[:path_cutoff_end]

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

                        # if VERBOSE and exp_number in PROBLEM_CASES:
                        if VERBOSE:
                            ###########################################################################
                            ########################### Signal plot ###################################
                            ###########################################################################
                            print(f'Exp. number: {exp_number}')
                            force_labels = ['fx', 'fy', 'fz']
                            fig_signals, axs_signals = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

                            for jdx in range(3):
                                axs_signals[0].plot(time, data[:, jdx], label=force_labels[jdx])

                            axs_signals[0].axvline(sig_start)
                            axs_signals[0].axvline(sig_stop)
                            axs_signals[1].plot(time, data[:, 3] * 1e+6) # to µm
                            axs_signals[1].plot(time, data[:, 4] * 1e+6) # to µm

                            axs_signals[2].contourf(time, freqs, power, cmap='inferno')
                            axs_signals[2].axhline(fz_freq*2)

                            axs_signals[3].plot(time, power[freq_div*2-1])

                            fig_signals.canvas.draw()

                            plt.setp(axs_signals[0].get_xticklabels(), visible=False)
                            plt.setp(axs_signals[1].get_xticklabels(), visible=False)
                            plt.setp(axs_signals[2].get_xticklabels(), visible=False)

                            axs_signals[0].set_ylabel('Force in N')
                            axs_signals[1].set_ylabel('Deflection in µm')
                            axs_signals[2].set_ylabel('Frequency in Hz')
                            axs_signals[3].set_ylabel('Wavelet intensity')
                            axs_signals[3].set_xlabel('Time in s')

                            fig_signals.align_ylabels()
                            fig_signals.tight_layout(pad=0.1)

                            ###########################################################################
                            ############################ Path plot ####################################
                            ###########################################################################
                            fig_path, axs_path = plt.subplots(1, 1, figsize=(8*CM_INCH, 8*CM_INCH))
                            # plot_path = axs_path.scatter(
                                # path[:, 0],
                                # path[:, 1],
                                # c=np.arange(0, len(path)),
                                # # c='#d95f02',
                                # s=1.5,
                                # cmap='viridis',
                                # label="Path"
                            # )
                            # cbar = fig_path.colorbar(plot_path, ax=axs_path, ticks=np.arange(0, len(path), 333))
                            # cbar.set_label('Path index', fontsize=FONTSIZE)
                            axs_path.plot(path[:, 0], path[:, 1], label="Path", color='#d95f02')

                            # circle_step = 500
                            # for path_idx in range(0, len(path), circle_step):
                                # axs_path.add_patch(
                                    # plt.Circle((path[path_idx, 0], path[path_idx, 1]), 4, color='b', fill=False)
                                # )

                            axs_path.add_patch(
                                plt.Circle((path[0, 0], path[0, 1]), 4, color='#7570b3', fill=False)
                            )
                            axs_path.add_patch(
                                plt.Circle((path[-1, 0], path[-1, 1]), 4, color='#7570b3', fill=False)
                            )

                            axs_path.add_patch(
                                Rectangle(
                                    (-34.5, -34.5),
                                    69,
                                    69,
                                    edgecolor='#1b9e77',
                                    facecolor='none',
                                    label='Workpiece'
                                )
                            )

                            axs_path.set_xlabel("x-coordinate", fontsize=FONTSIZE)
                            axs_path.set_ylabel("y-coordinate", fontsize=FONTSIZE)
                            fig_path.suptitle(
                                f'$n = {spsp}\,$rpm, '
                                f'$f_z = {fz}\,$mm, '
                                f'$a_e = {ae}\,$mm, '
                                f'$r_1 = {r1}°$, '
                                f'$r_2 = {r2}°$',
                                fontsize=FONTSIZE
                            )

                            axs_path.set_xticks(np.arange(-45, 46, 30))
                            axs_path.set_yticks(np.arange(-45, 46, 30))

                            fig_path.canvas.draw()

                            axs_path = modify_axis(axs_path, 'mm', 'mm', -2, -2, FONTSIZE, grid=False)

                            legend = axs_path.legend(
                                prop={'size':FONTSIZE},
                                bbox_to_anchor=(0., 1.04, 1., .102),
                                loc=3,
                                ncol=2,
                                borderaxespad=0.
                            )
                            legend.get_frame().set_linewidth(0.0)
                            legend.set_zorder(0)

                            axs_path.set_xlim(-45, 45)
                            axs_path.set_ylim(-45, 45)

                            fig_path.tight_layout(pad=0.1)

                        data = data[start_idx:stop_idx]
                        # dx = dx[start_idx:stop_idx]
                        # dy = dy[start_idx:stop_idx]

                        time = np.array([1 / f_s_downsamples * t_idx for t_idx in range(len(data))])

                        # if len(data) < len(path):
                            # path = path[:len(data)]
                            # path_time_up = path_time_up[:len(data)]
                        # else:
                            # data = data[:len(path)]
                            # time = time[:len(path)]

                        path_time = np.linspace(0, time[-1], len(path))
                        # jdx = 1
                        # while(path_time_up[-1] < path_time[-1]):
                            # path_time_up.append(1 / f_s_downsamples * jdx)
                            # jdx += 1

                        upsampled_path_x = np.interp(time, path_time, path[:, 0])
                        upsampled_path_y = np.interp(time, path_time, path[:, 1])

                        path = np.transpose([upsampled_path_x, upsampled_path_y])

                        overlap = np.array(
                            [ar_area(4, pos[0], pos[1], -34.5, -34.5, 34.5, 34.5) for pos in path]
                        )
                        overlap[overlap < 0] = 0

                        if VERBOSE:
                            ###########################################################################
                            ############################ Sync plot ####################################
                            ###########################################################################
                            fig_sync, axs_sync = plt.subplots(5, 1, figsize=(10, 10))#, sharex=True)
                            for jdx in range(3):
                                axs_sync[0].plot(time, data[:, jdx], label=force_labels[jdx])

                            axs_sync[1].plot(time, data[:, 3] * 1e+6) # to µm
                            axs_sync[1].plot(time, data[:, 4] * 1e+6) # to µm
                            axs_sync[2].plot(time, path[:, 0])
                            axs_sync[3].plot(time, path[:, 1])
                            axs_sync[4].plot(time, overlap)

                            fig_signals.canvas.draw()

                            # plt.setp(axs_sync[0].get_xticklabels(), visible=False)
                            # plt.setp(axs_sync[1].get_xticklabels(), visible=False)
                            # plt.setp(axs_sync[2].get_xticklabels(), visible=False)

                            axs_sync[0].set_ylabel('Force in N')
                            axs_sync[1].set_ylabel('Deflection in µm')
                            axs_sync[2].set_ylabel('x-coordinate in mm')
                            axs_sync[3].set_ylabel('y-coordinate im mm')
                            axs_sync[3].set_xlabel('Time in s')

                            fig_sync.align_ylabels()
                            fig_sync.tight_layout(pad=0.1)

                            plt.show()
                            plt.close()

                        features_target = np.c_[
                            time,
                            data[:, :3],
                            path,
                            overlap,
                            [spsp for __ in range(len(data))],
                            [fz for __ in range(len(data))],
                            [r1 for __ in range(len(data))],
                            [r2 for __ in range(len(data))],
                            data[:, 3:]
                            # dx,
                            # dy
                        ]
                        np.save(f'{PROCESSED_DIR}/{exp_number}_processed.npy', features_target)

                    scenarios.append(features_target)

            scenarios = np.array(scenarios, dtype=object)

            np.save(f'{PROCESSED_DIR}/processed_data.npy', scenarios)

        return scenarios
