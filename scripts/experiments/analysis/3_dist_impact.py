from pprint import pprint
import pandas as pd
import plotnine as p9

from neuralforecast.losses.numpy import smape
from config import PLOT_THEME

cv = pd.read_csv('assets/results/cv.csv')

pprint(cv.columns.tolist())

MODEL_NAMES = ['AutoKAN-P',
               'AutoMLP-P',
               'AutoNHITS-P',
               'AutoPatchTST-P',
               'AutoLSTM-P',
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
               'LGBM-P', ]

cv = cv[['unique_id', 'ds', 'cutoff', 'y'] + MODEL_NAMES]

cv_group = cv.groupby('unique_id')

results_by_series = {}
for g, df in cv_group:
    evaluation = {}
    for model in MODEL_NAMES:
        # evaluation[model] = rmae(y=df['y'], y_hat1=df[model], y_hat2=df['SNaive'])
        evaluation[model] = smape(y=df['y'], y_hat=df[model])

    results_by_series[g] = evaluation

results_df = pd.DataFrame(results_by_series).T

err = results_df.mean()

err_df = err.reset_index()
err_df.columns = ['model', 'value']

type_func_ = lambda x: 'Poisson' if x.endswith('-P') else ('Tweedie' if x.endswith('-T') else 'MAE')
err_df['type'] = err_df['model'].apply(type_func_)
err_df['model'] = err_df['model'].apply(lambda x: x.replace('-P', '').replace('-T', ''))

err_df.columns = ['Model', 'SMAPE', 'Objective']

plot = p9.ggplot(err_df, mapping=p9.aes(x='Model',
                                        y='SMAPE',
                                        fill='Objective')) + \
       p9.geom_bar(stat='identity',
                   position=p9.position_dodge(width=0.9),
                   width=0.8) + \
       p9.scale_fill_brewer(type='qual', palette='Set2') + \
       p9.labs(title='', x='', y='SMAPE') + \
       PLOT_THEME + \
       p9.theme(axis_text=p9.element_text(size=13))

plot.save('assets/outputs/plot_smape_dists.pdf', width=11, height=7)
