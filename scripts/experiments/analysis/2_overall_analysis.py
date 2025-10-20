import plotnine as p9
import pandas as pd

from config import PLOT_THEME

res_zim = pd.read_excel('assets/results/hfmd smape 14days.xlsx')
res_zim.columns = [
    'unique_id',
    'LM ZIP',
    'ZI Geometric',
    'ZI NB',
    'LM Poisson',
    'COM-Poisson'
]
res_zim['unique_id'] = res_zim['unique_id'].apply(lambda x: x.upper())
# res_zim = res_zim.drop(columns=['ZI Geometric','ZI NB','COM-Poisson'])

res = pd.read_csv('assets/outputs/results_by_series.csv')
res['unique_id'] = res['unique_id'].replace({'PP': 'PENANG', })

res = res.merge(res_zim, on='unique_id').set_index('unique_id')

# -------------------------

df = res.melt()

df.columns = ['Model', 'Error']
df_summary = df.groupby('Model').agg(['mean', 'std']).reset_index()
df_summary.columns = ['Model', 'Error', 'Error_std']
df_summary = df_summary.sort_values('Error', ascending=True)
df_summary['Model'] = pd.Categorical(df_summary['Model'].values.tolist(),
                                     categories=df_summary['Model'].values.tolist())

plot = \
    p9.ggplot(data=df_summary,
              mapping=p9.aes(x='Model',
                             y='Error')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='steelblue') + \
    p9.geom_errorbar(p9.aes(ymin='Error-Error_std',
                            ymax='Error+Error_std'),
                     width=0.2) + \
    PLOT_THEME + \
    p9.theme(axis_title_y=p9.element_text(size=11),
             axis_text=p9.element_text(size=11),
             axis_text_x=p9.element_text(angle=60)) + \
    p9.labs(x='', y='SMAPE')

plot.save('assets/outputs/plot_avg_error.pdf', width=11, height=7)

# error dist

ord = df_summary['Model']

df['Model'] = pd.Categorical(df['Model'].values.tolist(), categories=ord.values.tolist())

plot2 = p9.ggplot(df,
                  p9.aes(x='Model',
                         y='Error')) + \
        PLOT_THEME + \
        p9.geom_violin(width=0.9, fill='steelblue', show_legend=False) + \
        p9.coord_flip() + \
        p9.labs(x='', y='SMAPE') + \
        p9.guides(fill=None)

plot2.save('assets/outputs/plot_error_dist.pdf', width=11, height=7)

# error by loc

df_loc = res.T.melt()

df_loc.columns = ['Location', 'Error']
df_loc_summary = df_loc.groupby('Location').agg(['mean', 'std']).reset_index()
df_loc_summary.columns = ['Model', 'Error', 'Error_std']
df_loc_summary = df_loc_summary.sort_values('Error', ascending=True)
df_loc_summary['Model'] = pd.Categorical(df_loc_summary['Model'].values.tolist(),
                                         categories=df_loc_summary['Model'].values.tolist())

plot = \
    p9.ggplot(data=df_loc_summary,
              mapping=p9.aes(x='Model',
                             y='Error')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9,
                fill='orangered') + \
    p9.geom_errorbar(p9.aes(ymin='Error-Error_std',
                            ymax='Error+Error_std'),
                     width=0.2) + \
    PLOT_THEME + \
    p9.theme(axis_title_y=p9.element_text(size=11),
             axis_text=p9.element_text(size=11),
             axis_text_x=p9.element_text(angle=60)) + \
    p9.labs(x='', y='SMAPE')

plot.save('assets/outputs/plot_loc_avg_error.pdf', width=11, height=7)

# --- by horizon

res_hor = pd.read_csv('assets/outputs/results_by_horizon.csv')
res_hor = res_hor[['horizon','AutoNHITS','AutoMLP','AutoPatchTST','AutoKAN','LGBM','AutoLSTM']]

plot_hor = p9.ggplot(data=res_hor.melt('horizon'),
                    mapping=p9.aes(x='horizon',
                                   y='value',
                                   color='variable')) + \
          p9.geom_line() + \
          PLOT_THEME + \
          p9.labs(x='Horizon', y='SMAPE', color='variable') + \
          p9.theme(legend_position='right') + \
          p9.scale_x_continuous(breaks=range(1, max(res_hor['horizon'])+1))

plot_hor.save('assets/outputs/plot_error_horizon.pdf', width=12, height=6)


