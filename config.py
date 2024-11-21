import os

from pytorchutils.globals import nn

# Don't shit on lido
def available_cpu_count():
    """ Number of *available* virtual or physical CPUs on this system """
    # Tested with Python 3.3 - 3.13 on Linux
    try:
        res = len(os.sched_getaffinity(0))
        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

def div_counter(value, kernel_size, padding, stride, dilation):
    """
    Count the amount of necessary divisions based on model config,
    in order to reach 1
    """
    counter = 0
    while value > 1:
        value = (value + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        counter += 1
    return counter

def gen_channels(start_channels, div_count):
    """Generate list of amounts of channels for CNN based on div_count"""
    channels = [start_channels, 16]
    for idx in range(div_count - 1):
        channels.append(2**(5 + idx))
    return channels

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from scipy.stats import uniform, randint
import numpy as np

N_EDGES = 2

# Switches
VERBOSE = False
NN = True
CLUSTER_MODELING = True

config_str = 'new_defl_gmm8_force-path-overlay'

# Dirs
DATA_DIR = 'data/01_raw'
PROCESSED_DIR = 'data/02_processed'
MODEL_DIR = f'models/{config_str}'
RESULTS_DIR = f'results/{config_str}'
PLOT_DIR = f'plots/{config_str}'
PARAM_FILE = 'data/01_raw/clustersim_lhs_Zuordnung_Messdaten__FW.xlsx'

# Plot
OUT_LABELS = ['dx', 'dy']
FONTSIZE = 14

# Opt
CV_FOLDS = 5
N_ITER_SEARCH = 20
RANDOM_SEED = 1234

# Data props
TEST_SIZE = 0.2
INPUT_SIZE = 11
OUTPUT_SIZE = 2

PROBLEM_CASES = [
    16001,
    20002,
    21001,
    29001,
    30001,
    31001,
    36001,
    38001,
    40001,
    41001,
    44001,
    48001,
    49001,
    50001,
    58001,
    60002,
    61001,
    62003,
    64001
]

# Cluster stuff
N_CLUSTER = 8
N_CLUSTER_SILH = [3, 8, 12]
CLUSTER_COLS = [1, 2, 3, 4, 5, 6]
MIXTURE_TYPE = 'gmm'

############################
## Neural network shizzle ##
############################

# N_WINDOW = 6
N_WINDOW = INPUT_SIZE * 3
BATCH_SIZE = 1024

KERNEL_SIZE_CONV = 3
PADDING_CONV = 1
STRIDE_CONV = 1
DILATION_CONV = 1

KERNEL_SIZE_POOL = 2
PADDING_POOL = 0
STRIDE_POOL = 2
DILATION_POOL = 1

DIV_COUNT = div_counter(min(N_WINDOW, INPUT_SIZE), KERNEL_SIZE_POOL, PADDING_POOL, STRIDE_POOL, DILATION_POOL)-1
CHANNELS = gen_channels(1, DIV_COUNT)

model_config = {
    'input_size': [N_WINDOW, INPUT_SIZE],
    'random_seed': RANDOM_SEED,
    'models_dir': MODEL_DIR,
    'activation': 'ReLU', # Also tried PReLU and LeakyReLU
    'kernel_size_conv': KERNEL_SIZE_CONV,
    'padding_conv': PADDING_CONV,
    'stride_conv': STRIDE_CONV,
    'dilation_conv': DILATION_CONV,
    'kernel_size_pool': KERNEL_SIZE_POOL,
    'padding_pool': PADDING_POOL,
    'stride_pool': STRIDE_POOL,
    'dilation_pool':DILATION_POOL,
    'dimension': 2,
    'nb_layers': DIV_COUNT,
    # 'nb_units': 1014,
    'nb_units': None,
    'channels': CHANNELS,
    'batch_size': BATCH_SIZE,
    'output_size': OUTPUT_SIZE,
    'n_window': N_WINDOW,
    'init': 'kaiming_normal',
    'init_layers': (nn.Conv2d),
    'learning_rate': 0.0001, # Also tried 0.00001, was worse
    'max_iter': 401,
    'reg_lambda': 0.001,
    'dropout_rate': 0.0,
    'dropout_rate_conv': 0.0,
    # 'reg_lambda': 0,
    'loss': 'MSELoss',
    # Early stopper stuff
    'patience': 10,
    'max_problem': False,
}

############################
######## Ensembles #########
############################

PARAM_DICTS = [
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'max_depth': randint(2, 16),
        # 'subsample': uniform(0.5, 0.5),
        # 'n_estimators': randint(100, 250),
        # 'colsample_bytree': uniform(0.4, 0.6),
        # 'lambda': randint(1, 100),
        # 'gamma': uniform()
    # }
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'n_estimators': randint(100, 1000)
    # },
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'n_estimators': randint(100, 1000),
        # 'max_depth': randint(2, 32),
        # 'min_samples_split': randint(2, 11),
        # 'min_samples_leaf': randint(2, 11),
        # 'max_features': randint(1, INPUT_SIZE)
    # },
    {
        'n_estimators': randint(100, 250),
        'max_depth': randint(2, 16),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    }
]
REGRESSORS = [
    # [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(OUTPUT_SIZE)]
    # [AdaBoostRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    # [GradientBoostingRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [
        RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=available_cpu_count()
        ) for __ in range(OUTPUT_SIZE)
    ]
]
