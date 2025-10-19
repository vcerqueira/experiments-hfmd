from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,
    ARIMA,
    RandomWalkWithDrift,
    SimpleExponentialSmoothingOptimized,
    ADIDA,
    AutoTheta,
    CrostonOptimized,
    IMAPA,
    TSB,
    AutoETS
)

from src.data_reader import DataReader
from config import FREQ, PERIOD, TEST_SIZE, HORIZON, RESULTS_FILE_PATH

df = DataReader.load_data(drop_exogenous=True)

models = [
    RandomWalkWithDrift(),
    SeasonalNaive(season_length=PERIOD),
    AutoTheta(season_length=PERIOD),
    ARIMA(season_length=PERIOD, order=(5, 1, 1)),
    SimpleExponentialSmoothingOptimized(),
    AutoETS(),
    ADIDA(),
    CrostonOptimized(),
    IMAPA(),
    TSB(alpha_d=0.2, alpha_p=0.2),
]

sf = StatsForecast(
    models=models,
    freq=FREQ,
    n_jobs=1,
)

cv = sf.cross_validation(df=df, h=HORIZON, n_windows=TEST_SIZE)

cv = cv.reset_index()

cv.to_csv(f'{RESULTS_FILE_PATH}/cv_hfmd_cls.csv', index=False)
