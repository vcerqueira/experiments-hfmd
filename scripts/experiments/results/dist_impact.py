from pprint import pprint
import re
import pandas as pd
import plotnine as p9

from neuralforecast.losses.numpy import smape

cv_cls = pd.read_csv('assets/results/cv_classical.csv').drop(columns=['index'])
cv_nf = pd.read_csv('assets/results/cv_nf.csv')
cv_ml = pd.read_csv('assets/results/cv_mlf.csv')

cv = cv_nf.merge(cv_cls.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'])
cv = cv.merge(cv_ml.drop(columns=['y', 'index']), on=['unique_id', 'ds', 'cutoff'])
cv['ds'] = pd.to_datetime(cv['ds'])
# cv = cv.drop(columns=['LSTM(P)', 'NHITS(P)', 'TFT(P)',
#                       'LSTM(T)', 'NHITS(T)', 'TFT(T)',
#                       'LSTM(M)', 'NHITS(M)', 'TFT(M)',
#                       'LGBM', 'LGBM(tweedie)',
#                       'LSTM(T)-median', 'NHITS(T)-median', 'TFT(T)-median',
#                       'RWD', 'DT',
#                       'CrostonClassic', 'CrostonSBA'])

pprint(cv.columns.tolist())

meta_columns = 'y|unique_id|ds|cutoff|-hi-|-lo-'

model_names = [col for col in cv.columns if not re.search(meta_columns, col)]

cv = cv[['unique_id', 'ds', 'cutoff', 'y'] + model_names]

cv_group = cv.groupby('unique_id')

results_by_series = {}
for g, df in cv_group:
    evaluation = {}
    for model in model_names:
        # evaluation[model] = rmae(y=df['y'], y_hat1=df[model], y_hat2=df['SNaive'])
        evaluation[model] = smape(y=df['y'], y_hat=df[model])

    results_by_series[g] = evaluation

results_df = pd.DataFrame(results_by_series).T

err = results_df.mean()

err = err[['NHITS(M)', 'TFT(M)', 'LSTM(M)',
           'NHITS(P)-median', 'TFT(P)-median', 'LSTM(P)-median',
           'NHITS(T)-median', 'TFT(T)-median', 'LSTM(T)-median',
           'LGBM', 'LGBM(tweedie)', 'LGBM(poisson)']]

err = err.rename(lambda x: x.replace("-median", ""))
err = err.rename(lambda x: x.replace("tweedie", "T"))
err = err.rename(lambda x: x.replace("poisson", "P"))
err = err.rename(lambda x: 'LGBM(M)' if x == 'LGBM' else x)


def parse_index(idx):
    match = re.match(r'([A-Z]+)(?:\(([A-Z])\))?', idx)
    if match:
        model = match.group(1)
        type_val = match.group(2) if match.group(2) else 'None'
        return model, type_val
    return idx, 'None'


models = []
types = []
values = []

# Parse each index and value
for idx, value in err.items():
    model, type_val = parse_index(idx)
    models.append(model)
    types.append(type_val)
    values.append(value)

# Create DataFrame
result_df = pd.DataFrame({
    'Model': models,
    'Distribution': types,
    'SMAPE': values
})

result_df['Distribution'] = result_df['Distribution'].map({'M': 'MAE', 'P': 'Poisson', 'T': 'Tweedie'})

base_theme = p9.theme_538(base_family='Palatino', base_size=12) + \
             p9.theme(plot_margin=.025,
                      panel_background=p9.element_rect(fill='white'),
                      plot_background=p9.element_rect(fill='white'),
                      legend_box_background=p9.element_rect(fill='white'),
                      strip_background=p9.element_rect(fill='white'),
                      legend_background=p9.element_rect(fill='white'),
                      axis_text_x=p9.element_text(size=9, angle=0),
                      axis_text_y=p9.element_text(size=9),
                      legend_title=p9.element_blank())

plot = p9.ggplot(result_df, mapping=p9.aes(x='Model',
                                           y='SMAPE', fill='Distribution')) +\
        p9.geom_bar(stat='identity',
                    position=p9.position_dodge(width=0.9), width=0.8) +\
        p9.scale_fill_brewer(type='qual', palette='Set2') + \
        p9.labs(title='',
             x='',
             y='SMAPE') +\
        base_theme +\
        p9.theme(
            axis_text_x=p9.element_text(angle=0, size=12),
            plot_title=p9.element_text(hjust=0.5)
        )

plot.save('assets/outputs/plot_dists.pdf', width=11, height=7)
