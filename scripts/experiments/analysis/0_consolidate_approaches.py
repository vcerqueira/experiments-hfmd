import re

import pandas as pd

from src.data_reader import DataReader
from config import META_COLUMNS

cv_cls1 = pd.read_csv('assets/results/cv_hfmd_cls1.csv').drop(columns=['index'])
cv_cls2 = pd.read_csv('assets/results/cv_hfmd_cls2.csv').drop(columns=['index', 'RWD'])

cv_nf1 = pd.read_csv('assets/results/cv_hfmd_nf_poisson.csv')
cv_nf2 = pd.read_csv('assets/results/cv_hfmd_nf_mae.csv')
cv_nf3 = pd.read_csv('assets/results/cv_hfmd_nf_tw.csv')

BASE_COLUMNS = ['unique_id', 'y', 'ds', 'cutoff']

pattern_poisson = [c for c in cv_nf1.columns if c.startswith('Auto') and c.endswith('-median')]
cv_nf1 = cv_nf1[BASE_COLUMNS + pattern_poisson]
cv_nf1 = cv_nf1.rename(columns={col: col.replace('-median', '-P') for col in cv_nf1.columns})

pattern_tw = [c for c in cv_nf3.columns if c.startswith('Auto') and c.endswith('-median')]
cv_nf3 = cv_nf3[BASE_COLUMNS + pattern_tw]
cv_nf3 = cv_nf3.rename(columns={col: col.replace('-median', '-T') for col in cv_nf3.columns})

cv_ml = pd.read_csv('assets/results/cv_hfmd_mlf.csv')

cv = cv_cls1.merge(cv_cls2.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf1.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf2.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_nf3.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_ml.drop(columns=['y', 'index']), on=['unique_id', 'ds', 'cutoff'])
cv['ds'] = pd.to_datetime(cv['ds'])
cv['cutoff'] = pd.to_datetime(cv['cutoff'])

cv = DataReader.map_forecasting_horizon_col(cv)

model_names_map = {
    'LGBM(tweedie)': 'LGBM-T',
    'LGBM(poisson)': 'LGBM-P',
    'RF': 'RandomForest',
    'CrostonOptimized': 'Croston',
    'SESOpt': 'SES',
    'AutoETS': 'ETS',
}

cv = cv.rename(columns=model_names_map)

model_names = [col for col in cv.columns if not re.search(META_COLUMNS, col)]

cv = cv[['unique_id', 'ds', 'cutoff', 'horizon', 'y'] + model_names]

cv.to_csv('assets/results/cv.csv', index=False)
