import numpy as np
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from neuralforecast.losses.numpy import smape
from sklearn.model_selection import ParameterSampler

PARAMETER_SET = \
    dict(num_leaves=[3, 5, 10, 15, 30],
         max_depth=[-1, 3, 5, 10, 15],
         lambda_l1=[0, 0.1, 1, 100],
         lambda_l2=[0, 0.1, 1, 100],
         learning_rate=[0.05, 0.1, 0.2],
         min_child_samples=[15, 30, 50, 100],
         n_jobs=[1],
         linear_tree=[True, False],
         boosting_type=['gbdt'])


def optimize_lgb_mlf(df: pd.DataFrame,
                     frequency: str,
                     n_lags: int,
                     horizon: int,
                     test_size: int,
                     objective: str,
                     n_iter: int):

    param_list = list(ParameterSampler(PARAMETER_SET,
                                       n_iter=n_iter,
                                       random_state=np.random.RandomState(123)))

    param_loss = {}
    for i, params in enumerate(param_list):
        print(params)
        models = [lgb.LGBMRegressor(verbosity=-1,
                                    objective=objective,
                                    **params)]

        mlf = MLForecast(
            models=models,
            freq=frequency,
            lags=range(1, n_lags + 1),
        )

        cv = mlf.cross_validation(df=df,
                                  h=horizon,
                                  n_windows=test_size)

        iter_err = smape(y=cv['y'], y_hat=cv['LGBMRegressor'])

        param_loss[i] = iter_err

    best_idx = pd.Series(param_loss).argmin()

    best_params = param_list[best_idx]
    best_params['objective'] = objective

    return best_params
