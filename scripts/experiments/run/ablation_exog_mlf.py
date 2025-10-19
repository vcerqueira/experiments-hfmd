import lightgbm as lgb
from mlforecast import MLForecast
from neuralforecast.losses.numpy import smape

from codebase.data_reader import DataReader
from config import FREQ, TEST_SIZE, HORIZON, N_LAGS, POISSON_PARAMS

# TEST_SIZE = int(TEST_SIZE / 3)

df_exog = DataReader.load_data(drop_exogenous=False)
df_univ = DataReader.load_data(drop_exogenous=True)

models = {
    'LGBM': lgb.LGBMRegressor(verbosity=-1, **POISSON_PARAMS),
}

mlf = MLForecast(
    models=models,
    freq=FREQ,
    lags=range(1, N_LAGS + 1),
)

cv_mv = mlf.cross_validation(df=df_exog,
                             static_features=[],
                             h=HORIZON,
                             step_size=1,
                             n_windows=TEST_SIZE)
cv_uv = mlf.cross_validation(df=df_univ,
                             h=HORIZON,
                             step_size=1,
                             n_windows=TEST_SIZE)

print(smape(y=cv_mv['y'], y_hat=cv_mv['LGBM']))
print(smape(y=cv_uv['y'], y_hat=cv_uv['LGBM']))
# 0.5782932131956504
# 0.5782092504903217

