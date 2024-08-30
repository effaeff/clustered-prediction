from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from scipy.stats import uniform, randint

DATA_DIR = 'data/01_raw'
PROCESSED_DIR = 'data/02_processed'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
PLOT_DIR = 'plots'

FONTSIZE = 8

RANDOM_SEED = 1234

TEST_SIZE = 0.2
INPUT_SIZE = 8
OUTPUT_SIZE = 2

CV_FOLDS = 5
N_ITER_SEARCH = 20

VERBOSE = False

PARAM_DICTS = [
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'max_depth': randint(2, 32),
        # 'subsample': uniform(0.5, 0.5),
        # 'n_estimators': randint(100, 1000),
        # 'colsample_bytree': uniform(0.4, 0.6),
        # 'lambda': randint(1, 100),
        # 'gamma': uniform()
    # },
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'n_estimators': randint(100, 1000)
    # },
    {
        'learning_rate': uniform(0.0001, 0.1),
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    },
    {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    }
]
REGRESSORS = [
    # [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(OUTPUT_SIZE)],
    # [AdaBoostRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [GradientBoostingRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1) for __ in range(OUTPUT_SIZE)]
]
