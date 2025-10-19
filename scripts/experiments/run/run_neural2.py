from neuralforecast import NeuralForecast

from src.data_reader import DataReader
from src.nf_auto_models import get_auto_nf_models
from config import (FREQ, TEST_SIZE, HORIZON, RESULTS_FILE_PATH)

df = DataReader.load_data(drop_exogenous=True)

# df = df.head(3000)
# TEST_SIZE = 50

models = get_auto_nf_models(horizon=HORIZON,
                            loss='tweedie',
                            rs_n_samples=10)

nf = NeuralForecast(models=models, freq=FREQ)

cv = nf.cross_validation(df=df, n_windows=TEST_SIZE)
print(cv)
cv.to_csv(f'/Users/vcerq/Desktop/cv_hfmd_nf_tw.csv', index=False)
