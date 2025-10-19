import lightgbm as lgb
import xgboost as xgb
from mlforecast import MLForecast

from src.data_reader import DataReader
from config import (FREQ,
                    TEST_SIZE,
                    HORIZON,
                    N_LAGS,
                    POISSON_PARAMS,
                    REGRESSION_PARAMS,
                    TWEEDIE_PARAMS,
                    RESULTS_FILE_PATH)

df = DataReader.load_data(drop_exogenous=True)
# df = df.head(1000)

models = {
    'LGBM': lgb.LGBMRegressor(verbosity=-1, **REGRESSION_PARAMS),
    'LGBM(tweedie)': lgb.LGBMRegressor(verbosity=-1, **TWEEDIE_PARAMS),
    'LGBM(poisson)': lgb.LGBMRegressor(verbosity=-1, **POISSON_PARAMS),
    'RF': xgb.XGBRFRegressor(n_estimators=50),
}

mlf = MLForecast(
    models=models,
    freq=FREQ,
    lags=range(1, N_LAGS + 1),
)

cv = mlf.cross_validation(df=df,
                          h=HORIZON,
                          step_size=1,
                          n_windows=TEST_SIZE)

cv = cv.reset_index()

cv.to_csv(f'/Users/vcerq/Desktop/cv_hfmd_mlf.csv', index=False)
