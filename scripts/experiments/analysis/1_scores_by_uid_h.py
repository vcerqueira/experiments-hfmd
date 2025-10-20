from pprint import pprint
import re

import pandas as pd
from neuralforecast.losses.numpy import smape

from config import META_COLUMNS

cv = pd.read_csv('assets/results/cv.csv')

# pprint(cv.columns.tolist())
MODELS_TO_DROP = [
    'AutoKAN',
    'AutoMLP',
    'AutoNHITS',
    'AutoPatchTST',
    'AutoLSTM',
    'AutoKAN-T',
    'AutoMLP-T',
    'AutoNHITS-T',
    'AutoPatchTST-T',
    'AutoLSTM-T',
    'LGBM',
    'LGBM-T',
    'RandomForest'
]

cv = cv.drop(columns=MODELS_TO_DROP)
cv = cv.rename(columns={col: col.replace('-P', '') for col in cv.columns})

model_names = [col for col in cv.columns if not re.search(META_COLUMNS, col)]

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
