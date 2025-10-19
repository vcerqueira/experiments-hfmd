import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.data_reader import DataReader
from config import FREQ, PERIOD

df = DataReader.load_data(drop_exogenous=True)

train, _ = DataReader.train_test_split(df, 365 * 6)
# 5 years for model selection
print(train.shape)


def get_arima_order(mod):
    order = tuple(mod["arma"][i] for i in [0, 5, 1, 2, 6, 3, 4])

    ord = pd.Series(order, index=['AR', 'I', 'MA', 'S_AR', 'S_I', 'S_MA', 'm'])

    alias = f'ARIMA({ord[0]},{ord[1]},{ord[2]})({ord[3]},{ord[4]},{ord[5]})[365]'
    return alias


models = [
    AutoARIMA(season_length=PERIOD, trace=True),
]

sf = StatsForecast(
    models=models,
    freq=FREQ,
    n_jobs=1,
)

sf.fit(df=train)

best_configs = pd.Series([get_arima_order(mod.model_) for mod in sf.fitted_.flatten()])
print(best_configs.value_counts())
