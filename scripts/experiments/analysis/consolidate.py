from pprint import pprint
import re

import pandas as pd
from neuralforecast.losses.numpy import smape

from src.data_reader import DataReader

cv_cls1 = pd.read_csv('assets/results/cv_hfmd_cls1.csv').drop(columns=['index'])
cv_cls2 = pd.read_csv('assets/results/cv_hfmd_cls2.csv').drop(columns=['index','RWD'])

cv_nf1 = pd.read_csv('assets/results/cv_hfmd_nf_poisson.csv')
cv_nf2 = pd.read_csv('assets/results/cv_hfmd_nf_mae.csv')
cv_nf3 = pd.read_csv('assets/results/cv_hfmd_nf_tw.csv')
cv_nf1 = cv_nf1.drop(columns=[col for col in cv_nf1.columns if '-hi-' in col or '-lo-' in col])
cv_nf2 = cv_nf2.drop(columns=[col for col in cv_nf2.columns if '-hi-' in col or '-lo-' in col])
cv_nf3 = cv_nf3.drop(columns=[col for col in cv_nf3.columns if '-hi-' in col or '-lo-' in col])

cv_ml = pd.read_csv('assets/results/cv_hfmd_mlf.csv')


cv = cv_cls1.merge(cv_cls2.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf1.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf2.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf3.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_ml.drop(columns=['y', 'index']), on=['unique_id', 'ds', 'cutoff'])



cv = cv_nf.merge(cv_cls.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_ml.drop(columns=['y', 'index']), on=['unique_id', 'ds', 'cutoff'])
cv['ds'] = pd.to_datetime(cv['ds'])

cv = cv.drop(columns=['LSTM(P)', 'NHITS(P)', 'TFT(P)',
                      'LSTM(T)', 'NHITS(T)', 'TFT(T)',
                      'LSTM(M)', 'NHITS(M)', 'TFT(M)',
                      'LGBM', 'LGBM(tweedie)',
                      'LSTM(T)-median', 'NHITS(T)-median', 'TFT(T)-median',
                      'RWD', 'DT',
                      'CrostonClassic', 'CrostonSBA'])

pprint(cv.columns.tolist())

cv = DataReader.map_forecasting_horizon_col(cv)

meta_columns = 'y|unique_id|ds|cutoff|-hi-|-lo-|horizon'

model_names_map = {
    'LSTM(P)-median': 'LSTM',
    'NHITS(P)-median': 'NHITS',
    'TFT(P)-median': 'TFT',
    'RandomForestRegressor': 'RF',
    'LGBM(poisson)': 'LGBM',
    'CrostonOptimized': 'Croston',
    'SESOpt': 'SES',
    'AutoETS': 'ETS',
    'SeasonalNaive': 'SNaive',
}

cv = cv.rename(columns=model_names_map)

model_names = [col for col in cv.columns if not re.search(meta_columns, col)]

cv = cv[['unique_id', 'ds', 'cutoff', 'horizon', 'y'] + model_names]

# evaluate(df=cv, metrics=[smape],models=model_names)

cv.to_csv('assets/outputs/cv.csv', index=False)

# error by series
cv_group = cv.groupby('unique_id')

results_by_series = {}
for g, df in cv_group:
    evaluation = {}
    for model in model_names:
        # evaluation[model] = rmae(y=df['y'], y_hat1=df[model], y_hat2=df['SNaive'])
        evaluation[model] = smape(y=df['y'], y_hat=df[model])

    results_by_series[g] = evaluation

results_df = pd.DataFrame(results_by_series).T

results_df.to_csv('assets/outputs/results_by_series.csv', index_label='unique_id')

print(results_df.median().sort_values())
print(results_df.mean().sort_values())

# by horizon

cv_group = cv.groupby('horizon')

# evaluate(df=cv, metrics=[smape], models=model_names)

results_by_horizon = {}
for g, df in cv_group:
    evaluation = {}
    for model in model_names:
        # evaluation[model] = rmae(y=df['y'], y_hat1=df[model], y_hat2=df['SNaive'])
        evaluation[model] = smape(y=df['y'], y_hat=df[model])

    results_by_horizon[g] = evaluation

results_df = pd.DataFrame(results_by_horizon).T

results_df.to_csv('assets/outputs/results_by_horizon.csv', index_label='horizon')

print(results_df.median().sort_values())
print(results_df.mean().sort_values())
