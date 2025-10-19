from pprint import pprint

from src.param_optim import optimize_lgb_mlf
from src.data_reader import DataReader
from config import FREQ, HORIZON, N_LAGS, TEST_SIZE, PERIOD

df = DataReader.load_data(drop_exogenous=True)
train, _ = DataReader.train_test_split(df, TEST_SIZE)

params = optimize_lgb_mlf(df=train,
                          frequency=FREQ,
                          horizon=HORIZON,
                          n_iter=50,
                          n_lags=N_LAGS,
                          objective='poisson',
                          test_size=30)

pprint(params)
