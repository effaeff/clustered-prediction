"""Learning routine"""

import numpy as np
import sys

import os
from joblib import dump, load

from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV

import warnings

from config import (
    MODEL_DIR,
    REGRESSORS,
    PARAM_DICTS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CV_FOLDS,
    N_ITER_SEARCH,
    CLUSTER_MODELING,
    N_CLUSTER,
    CLUSTER_COLS,
    OUT_LABELS,
    VERBOSE
)

def fit(inp, target, regressor, param_dict, hyperopt_fname):
    """Fitting method"""
    if os.path.isfile(hyperopt_fname):
        rand_search = load(hyperopt_fname)
    else:
        print(f'Fitting {hyperopt_fname}...')
        rand_search = RandomizedSearchCV(
            regressor,
            param_distributions=param_dict,
            n_iter=N_ITER_SEARCH,
            cv=CV_FOLDS,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,
            error_score='raise',
            verbose=(4 if VERBOSE else 0)
        )
        rand_search.fit(
            inp,
            target
        )
        dump(rand_search, hyperopt_fname)
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
        for reg_idx in range(len(REGRESSORS)):
            for out_idx in range(OUTPUT_SIZE):
                if CLUSTER_MODELING:
                    for cluster_idx in range(N_CLUSTER):
                        cluster_data = train_data[train_data[:, -1]==cluster_idx, :-1]
                        if len(cluster_data) > 0:
                            inp = cluster_data[:, :INPUT_SIZE]
                            target = cluster_data[:, INPUT_SIZE + out_idx]
                            hyperopt_fname = (
                                f'{MODEL_DIR}/hyperopt_{REGRESSORS[reg_idx][0].__class__.__name__}'
                                f'_{OUT_LABELS[out_idx]}_cluster-{cluster_idx}.joblib'
                            )
                            hyperopts[reg_idx, out_idx, cluster_idx] = fit(
                                inp,
                                target,
                                REGRESSORS[reg_idx][out_idx],
                                PARAM_DICTS[reg_idx],
                                hyperopt_fname
                            )
                else:
                    inp = train_data[:, :INPUT_SIZE]
                    target = train_data[:, INPUT_SIZE + out_idx]
                    hyperopt_fname = (
                        f'{MODEL_DIR}/hyperopt_{REGRESSORS[reg_idx][0].__class__.__name__}'
                        f'_{OUT_LABELS[out_idx]}.joblib'
                    )
                    hyperopts[reg_idx, out_idx] = fit(
                        inp,
                        target,
                        REGRESSORS[reg_idx][out_idx],
                        PARAM_DICTS[reg_idx],
                        hyperopt_fname
                    )
    return hyperopts
