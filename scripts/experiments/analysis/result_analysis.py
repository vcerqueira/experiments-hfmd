import plotnine as p9
import pandas as pd

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
res_zim = res_zim.drop(columns=['ZI Geometric','ZI NB','COM-Poisson'])

res = pd.read_csv('assets/outputs/results_by_series.csv')
res['unique_id'] = res['unique_id'].replace({'PP': 'PENANG', })

res = res.merge(res_zim, on='unique_id').set_index('unique_id')

# -------------------------
# -------------------------


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

#

df = res.melt()

df.columns = ['Model', 'Error']
df_summary = df.groupby('Model').mean().reset_index().sort_values('Error', ascending=False)
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
    base_theme + \
    p9.theme(axis_title_y=p9.element_text(size=11),
             axis_text=p9.element_text(size=11)) + \
    p9.labs(x='', y='SMAPE') + \
    p9.coord_flip()  # + \
# p9.scale_fill_manual(values=cls.COLOR_MAP) + \
# p9.guides(fill=None)

plot.save('assets/outputs/plot_avg_error.pdf', width=11, height=7)
# error dist

ord = df_summary['Model']
ord.values.tolist()

df['Model'] = pd.Categorical(df['Model'].values.tolist(), categories=ord.values.tolist())

plot2 = p9.ggplot(df,
                  p9.aes(x='Model',
                         y='Error')) + \
        base_theme + \
        p9.geom_boxplot(
            width=0.9,
            fill='steelblue',
            show_legend=False) + \
        p9.coord_flip() + \
        p9.labs(x='Error distribution', y='SMAPE') + \
        p9.guides(fill=None)

plot2.save('assets/outputs/plot_dist.pdf', width=11, height=7)
