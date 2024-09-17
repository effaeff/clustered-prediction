"""Learning routine"""

import numpy as np
import sys
import os
default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV

import warnings

from config import (
    REGRESSORS,
    PARAM_DICTS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CV_FOLDS,
    N_ITER_SEARCH,
    CLUSTER_MODELING,
    N_CLUSTER,
    CLUSTER_COLS
)

def fit(inp, target, regressor, param_dict):
    """Fitting method"""
    rand_search = RandomizedSearchCV(
        regressor,
        param_distributions=param_dict,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        error_score='raise'
    )
    rand_search.fit(
        inp,
        target
    )
    return rand_search

def train(train_data):
    """Learning method"""
    if not sys.warnoptions:
        # warnings.simplefilter("ignore")
        # os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
        hyperopts = (
            np.empty((len(REGRESSORS), OUTPUT_SIZE, N_CLUSTER), dtype=object) if CLUSTER_MODELING else
            np.empty((len(REGRESSORS), OUTPUT_SIZE), dtype=object)
        )
        for reg_idx in tqdm(range(len(REGRESSORS))):
            for out_idx in range(OUTPUT_SIZE):
                if CLUSTER_MODELING:
                    for cluster_idx in range(N_CLUSTER):
                        cluster_data = train_data[train_data[:, -1]==cluster_idx, :-1]
                        inp = cluster_data[:, :INPUT_SIZE]
                        target = cluster_data[:, INPUT_SIZE + out_idx]
                        hyperopts[reg_idx, out_idx, cluster_idx] = fit(
                            inp,
                            target,
                            REGRESSORS[reg_idx][out_idx],
                            param_dict=PARAM_DICTS[reg_idx]
                        )
                else:
                    inp = train_data[:, :INPUT_SIZE]
                    target = train_data[:, INPUT_SIZE + out_idx]
                    hyperopts[reg_idx, out_idx] = fit(
                        inp,
                        target,
                        REGRESSORS[reg_idx][out_idx],
                        param_dict=PARAM_DICTS[reg_idx]
                    )
    return hyperopts
