"""Script for process clustering"""

import os
default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import misc
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import MinMaxScaler

from force2defl.data_processing import DataProcessing
from force2defl.nn_trainer import Trainer, ClusterTrainer
from force2defl.clusterer import Clusterer
from force2defl.train import train
# from force2defl.tcn import TCN
from pytorchutils.cnn import CNNModel
from force2defl.test import test
from force2defl.utils import write_results, load_estimators

from config import (
    MODEL_DIR,
    PLOT_DIR,
    RESULTS_DIR,
    PROCESSED_DIR,
    REGRESSORS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CLUSTER_MODELING,
    N_CLUSTER,
    CLUSTER_COLS,
    NN,
    model_config
)

def main():
    """Main method"""
    misc.gen_dirs([MODEL_DIR, PLOT_DIR, RESULTS_DIR, PROCESSED_DIR])
    processing = DataProcessing()

    if NN:
        if CLUSTER_MODELING:
            trainer = ClusterTrainer(model_config, processing)
            # trainer.train(validate_every=10, save_every=10, save_eval=True, verbose=True)
            trainer.validate(save_eval=False)
        else:
            model = CNNModel(model_config)
            trainer = Trainer(model_config, model, processing)
            trainer.get_batches_fn = processing.get_batches
            # trainer.train(validate_every=10, save_every=10, save_eval=True, verbose=True)
            trainer.validate(-1, False, True, '')
    else:
        train_data, test_data = processing.get_train_test()
        hyperopts = train(train_data)
        total_errors = np.empty((len(hyperopts), OUTPUT_SIZE))
        total_variances = np.empty((len(hyperopts), OUTPUT_SIZE))
        for hyper_idx, hyperopt in enumerate(hyperopts):
            errors, variances = test(hyperopt, test_data)
            total_errors[hyper_idx] = errors
            total_variances[hyper_idx] = variances
        write_results(hyperopts, total_errors, total_variances)

if __name__ == '__main__':
    main()
