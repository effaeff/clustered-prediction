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
    RESULTS_DIR
)

def test(hyperopt, test_data):
    errors = np.zeros(OUTPUT_SIZE)
    variances = np.zeros(OUTPUT_SIZE)

    for scenario_idx, test_scenario in enumerate(test_data):

        total_pred = np.empty((len(test_scenario), OUTPUT_SIZE))
        total_target = test_scenario[:, INPUT_SIZE:]
        for out_idx in range(OUTPUT_SIZE):
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
