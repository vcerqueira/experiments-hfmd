import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from mlforecast import MLForecast
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from mlforecast.auto import AutoMLForecast

from mlforecast.auto import (AutoLasso,
                             AutoRidge,
                             AutoXGBoost,
                             AutoLightGBM,
                             AutoElasticNet)

from src.data_reader import DataReader
from config import (FREQ,
                    TEST_SIZE,
                    HORIZON, PERIOD,
                    N_LAGS,
                    POISSON_PARAMS,
                    REGRESSION_PARAMS,
                    TWEEDIE_PARAMS,
                    RESULTS_FILE_PATH)

df = DataReader.load_data(drop_exogenous=True)
df = df.head(1000)


def lightgbm_space_l1l2(trial: optuna.Trial):
    return {
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "objective": trial.suggest_categorical("objective", ["l1", "l2", "tweedie", "poisson"]),
    }


def lightgbm_space_tweedie(trial: optuna.Trial):
    return {
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "objective": trial.suggest_categorical("objective", ["tweedie"]),
    }


def lightgbm_space_poisson(trial: optuna.Trial):
    return {
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "objective": trial.suggest_categorical("objective", ["poisson"]),
    }


def get_amlf_models():
    auto_models_ml = {
        'LGB(L1/L2)': AutoLightGBM(config=lightgbm_space_l1l2),
        'LGB(T)': AutoLightGBM(config=lightgbm_space_tweedie),
        'LGB(P)': AutoLightGBM(config=lightgbm_space_poisson),
        'Ridge': AutoRidge(),
        'Lasso': AutoLasso(),
    }

    return auto_models_ml


from mlforecast.target_transforms import LocalMinMaxScaler


def my_init_config(trial):
    return {
        # 'target_transforms': [LocalMinMaxScaler()],
        'target_transforms': [],
    }


mlf = AutoMLForecast(
    models=get_amlf_models(),
    freq=FREQ,
    season_length=PERIOD,
    init_config=my_init_config

)

TEST_SIZE = 50


mlf.fit(df=df,
        h=HORIZON,
        step_size=1,
        n_windows=TEST_SIZE,
        num_samples=2)

# 1. Fit AutoMLForecast (performs CV internally for optimization)
automlf.fit(df=df, h=HORIZON, n_windows=TEST_SIZE, num_samples=20)

# 2. Access the best model and run cross_validation
best_model = automlf.models_['model_name']  # e.g., 'AutoLightGBM'
cv_results = best_model.cross_validation(df=df, h=HORIZON, step_size=1, n_windows=TEST_SIZE)

cv = mlf.cross_validation(df=df,
                          h=HORIZON,
                          step_size=1,
                          n_windows=TEST_SIZE)

cv = cv.reset_index()

cv.to_csv(f'{RESULTS_FILE_PATH}/cv_mlf.csv', index=False)


##
import os
import tempfile

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from utilsforecast.plotting import plot_series

from mlforecast import MLForecast
from mlforecast.auto import (
    AutoLightGBM,
    AutoMLForecast,
    AutoModel,
    AutoRidge,
    ridge_space,
)
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean

def my_lgb_config(trial: optuna.Trial):
    return {
        'learning_rate': 0.05,
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 2, 128, log=True),
        'objective': trial.suggest_categorical('objective', ['tweedie']),
    }

my_lgb = AutoModel(
    model=lgb.LGBMRegressor(),
    config=my_lgb_config,
)
auto_mlf = AutoMLForecast(
    models={'my_lgb': my_lgb},
    freq='D',
    season_length=24,
).fit(
    df,
    n_windows=2,
    h=5,
    num_samples=2,
)
preds = auto_mlf.predict(horizon)

