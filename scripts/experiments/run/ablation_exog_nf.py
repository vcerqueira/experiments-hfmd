from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss

from codebase.data_reader import DataReader
from config import FREQ, TEST_SIZE, NHITS_CONFIG
from neuralforecast.losses.numpy import smape

df_exog = DataReader.load_data(drop_exogenous=False)

poisson_loss = DistributionLoss(distribution='Poisson', level=[80, 90], return_params=False)

models = [
    NHITS(loss=poisson_loss, **NHITS_CONFIG, alias='NHITS'),
    NHITS(loss=poisson_loss, **NHITS_CONFIG,
          hist_exog_list=['temperature', 'rainfall', 'humidity'], alias='NHITS(Exog)'),
]

nf = NeuralForecast(models=models, freq=FREQ)

cv = nf.cross_validation(df=df_exog, n_windows=TEST_SIZE)
cv = cv.reset_index()

print(smape(y=cv['y'], y_hat=cv['NHITS-median']))
print(smape(y=cv['y'], y_hat=cv['NHITS(Exog)-median']))
# 0.5610176578032522
# 0.5634947914768041

# {
#     'LGBM-Univariate': 0.5782092504903217,
#     'LGBM-Exogenous': 0.5782932131956504,
#     'NHITS-Univariate': 0.5610176578032522,
#     'NHITS-Exogenous': 0.5634947914768041,
# }