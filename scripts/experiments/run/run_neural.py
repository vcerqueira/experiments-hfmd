from neuralforecast import NeuralForecast

from src.data_reader import DataReader
from src.nf_auto_models import get_auto_nf_models
from config import (FREQ, TEST_SIZE, HORIZON, RESULTS_FILE_PATH)

df = DataReader.load_data(drop_exogenous=True)

models = get_auto_nf_models(horizon=HORIZON,
                            loss='tweedie',
                            rs_n_samples=10)

nf = NeuralForecast(models=models, freq=FREQ)

cv = nf.cross_validation(df=df, n_windows=TEST_SIZE)
cv.to_csv(f'{RESULTS_FILE_PATH}/cv_hfmd_nf_tw.csv', index=False)
