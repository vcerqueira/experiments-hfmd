import pandas as pd
from neuralforecast import NeuralForecast

from src.data_reader import DataReader
from src.nf_auto_models import get_auto_nf_models
from config import (FREQ, TEST_SIZE, HORIZON, RESULTS_FILE_PATH)

df = DataReader.load_data(drop_exogenous=True)

models = get_auto_nf_models(horizon=HORIZON,
                            loss='poisson',
                            rs_n_samples=10)

nf = NeuralForecast(models=models, freq=FREQ)

cv_results = []
for group_name, group_df in df.groupby('unique_id'):
    cv = nf.cross_validation(df=group_df, n_windows=TEST_SIZE)
    cv['unique_id'] = group_name
    cv_results.append(cv)

cv_combined = pd.concat(cv_results, axis=0)
cv_combined.to_csv(f'{RESULTS_FILE_PATH}/cv_hfmd_nf_local.csv', index=False)
