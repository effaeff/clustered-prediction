"""Testing routine using trained regressor"""

import re
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from plot_utils import modify_axis
import matplotlib.ticker as mticker
from matplotlib import rc
rc('font', family='Arial')

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    FONTSIZE,
    PLOT_DIR,
    RESULTS_DIR,
    CLUSTER_MODELING,
    N_CLUSTER
)

def test(hyperopt, test_data):
    errors = np.zeros(OUTPUT_SIZE)
    variances = np.zeros(OUTPUT_SIZE)
    np.set_printoptions(suppress=True)

    for scenario_idx, test_scenario in enumerate(test_data):
        total_pred = np.empty((len(test_scenario), OUTPUT_SIZE))
        total_target = test_scenario[:, INPUT_SIZE:]
        for out_idx in range(OUTPUT_SIZE):
            if CLUSTER_MODELING:
                pred = np.array([])
                for cluster_idx in range(N_CLUSTER):
                    cluster_data = test_scenario[test_scenario[:, -1]==cluster_idx, :-1]
                    cluster_pred = hyperopt[out_idx, cluster_idx].predict(cluster_data[:, :INPUT_SIZE])

                    timed_pred = np.c_[cluster_data[:, 0], cluster_pred]
                    pred = np.vstack([pred, timed_pred]) if pred.size else timed_pred

                    # pred = np.concatenate((pred, np.c_[cluster_data[:, 0], pred]))

                # Sort cluster-based prediction according to time channel
                pred = pred[pred[:, 0].argsort(), 1]

            else:
                pred = hyperopt[out_idx].predict(test_scenario[:, :INPUT_SIZE])

            target = total_target[:, out_idx]
            total_pred[:, out_idx] = pred

            errors[out_idx] = math.sqrt(
                mean_squared_error(target, pred)
            ) / np.ptp(target) * 100.0
            variances[out_idx] = np.std(
                [
                    abs(target[idx] - pred[idx]) / np.ptp(target)
                    for idx in range(len(target))
                ]
            ) * 100.0

        fig, axs = plt.subplots(2, 1, sharex=True)
        for idx, ax in enumerate(axs):
            ax.plot(total_target[:, idx], label='Target')
            ax.plot(total_pred[:, idx], label='Prediction')


        axs[0].legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc='lower left',
            ncol=2,
            mode="expand",
            fontsize=FONTSIZE,
            borderaxespad=0.,
            frameon=False
        )
        plt.savefig(
            f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}_scenario{scenario_idx}.png',
            dpi=600
        )
        # plt.show()

    return errors, variances
