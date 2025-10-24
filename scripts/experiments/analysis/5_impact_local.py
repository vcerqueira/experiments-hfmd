from pprint import pprint

import numpy as np
import pandas as pd
import plotnine as p9

from neuralforecast.losses.numpy import smape
from config import PLOT_THEME

cv_uv = pd.read_csv('assets/results/cv.csv')
cv_loc = pd.read_csv('assets/results/cv_hfmd_nf_local.csv')

MODEL_NAMES = ['AutoKAN-P', 'AutoMLP-P', 'AutoNHITS-P', 'AutoPatchTST-P', 'AutoLSTM-P']
MODEL_NAMES_LOC = ['AutoKAN-median', 'AutoMLP-median', 'AutoNHITS-median', 'AutoPatchTST-median', 'AutoLSTM-median']
META_COLS = ['unique_id', 'ds', 'cutoff', 'y']

cv_uv = cv_uv[META_COLS + MODEL_NAMES]
# cv = cv.rename(columns={col: col.replace('-P', '') for col in cv.columns})
cv_loc = cv_loc[META_COLS + MODEL_NAMES_LOC]
cv_loc = cv_loc.rename(columns={col: col.replace('-median', '') for col in cv_loc.columns})

pprint(cv_uv.columns.tolist())
pprint(cv_loc.columns.tolist())

cv = cv_uv.merge(cv_loc.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
pprint(cv.columns.tolist())

all_models = [c for c in cv.columns if c not in META_COLS]

cv_group = cv.groupby('unique_id')

results_by_series = {}
for g, df in cv_group:
    evaluation = {}
    for model in all_models:
        # evaluation[model] = rmae(y=df['y'], y_hat1=df[model], y_hat2=df['SNaive'])
        evaluation[model] = smape(y=df['y'], y_hat=df[model])

    results_by_series[g] = evaluation

results_df = pd.DataFrame(results_by_series).T

err = results_df.mean()

err_df = err.reset_index()
err_df.columns = ['model', 'value']

err_df['type'] = err_df['model'].apply(lambda x: 'Global' if x.endswith('-P') else 'Local')
err_df['model'] = err_df['model'].apply(lambda x: x.replace('-P', ''))

err_df.columns = ['Model', 'SMAPE', 'Approach']

plot = p9.ggplot(err_df, mapping=p9.aes(x='Model',
                                        y='SMAPE',
                                        fill='Approach')) + \
       p9.geom_bar(stat='identity',
                   position=p9.position_dodge(width=0.9),
                   width=0.8) + \
       p9.scale_fill_brewer(type='qual', palette='Set2') + \
       p9.labs(title='', x='', y='SMAPE') + \
       PLOT_THEME + \
       p9.theme(axis_text=p9.element_text(size=13))

plot.save('assets/outputs/plot_smape_local.pdf', width=11, height=7)
