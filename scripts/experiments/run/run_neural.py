from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT, LSTM
from neuralforecast.losses.pytorch import DistributionLoss, MAE

from codebase.data_reader import DataReader
from config import FREQ, TEST_SIZE, HORIZON, N_LAGS, RESULTS_FILE_PATH, TFT_CONFIG, LSTM_CONFIG, NHITS_CONFIG

df = DataReader.load_data(drop_exogenous=True)
# df=df.head(3000)

poisson_loss = DistributionLoss(distribution='Poisson', level=[80, 90], return_params=False)
tweedie_loss = DistributionLoss(distribution='Tweedie', level=[80, 90], rho=1.5, return_params=False)
mae_loss = MAE()

config_base = {'h': HORIZON, 'input_size': N_LAGS}
config_mae = {'loss': mae_loss}
config_poisson = {'loss': poisson_loss}
config_tweedie = {'loss': tweedie_loss}

# TFT_CONFIG['max_steps'] = 10
# LSTM_CONFIG['max_steps'] = 10
# NHITS_CONFIG['max_steps'] = 10

models = [
    NHITS(**NHITS_CONFIG, **config_mae, alias='NHITS(M)'),
    TFT(**TFT_CONFIG, **config_mae, alias='TFT(M)'),
    LSTM(**LSTM_CONFIG, **config_mae, alias='LSTM(M)'),

    NHITS(**NHITS_CONFIG, **config_poisson, alias='NHITS(P)'),
    TFT(**TFT_CONFIG, **config_poisson, alias='TFT(P)'),
    LSTM(**LSTM_CONFIG, **config_poisson, alias='LSTM(P)'),

    NHITS(**NHITS_CONFIG, **config_tweedie, alias='NHITS(T)'),
    TFT(**TFT_CONFIG, **config_tweedie, alias='TFT(T)'),
    LSTM(**LSTM_CONFIG, **config_tweedie, alias='LSTM(T)'),
]

nf = NeuralForecast(models=models, freq=FREQ)

cv = nf.cross_validation(df=df, n_windows=TEST_SIZE)
cv.to_csv(f'{RESULTS_FILE_PATH}/cv_nf.csv', index=False)
