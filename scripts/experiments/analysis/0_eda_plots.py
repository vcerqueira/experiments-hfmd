import numpy as np
import pandas as pd
import plotnine as p9

from src.data_reader import DataReader
from config import PLOT_THEME

df = DataReader.load_data()

df_ = df.drop(['unique_id', 'ds'], axis=1)

corr_df = df_.corr()
# corr_df = df.corr(method='kendall')

dfm = corr_df.reset_index().melt('index')
dfm['index'] = dfm['index'].apply(lambda x: x.capitalize()).replace({'Y': 'HMFD'})
dfm['variable'] = dfm['variable'].apply(lambda x: x.capitalize()).replace({'Y': 'HMFD'})
dfm.columns = ['v1', 'v2', 'Correlation']

# correlation heatmap

plot = \
    p9.ggplot(data=dfm,
              mapping=p9.aes(x='v1',
                             y='v2', fill='Correlation')) + \
    p9.geom_tile(p9.aes(width=0.95, height=0.95)) + \
    PLOT_THEME + \
    p9.theme(axis_title=p9.element_blank(),
             axis_text=p9.element_text(size=12)) + \
    p9.scale_fill_gradient2(low='red', mid='white', high='blue')

plot.save('assets/outputs/plot_eda_corr.pdf', width=6, height=6)

#

df_uv = df.drop(columns=['temperature', 'rainfall', 'humidity'])
df_uv_m = df_uv.groupby('unique_id').resample('MS', on='ds').agg({'y': 'sum'}).reset_index()

plot_uv = p9.ggplot(data=df_uv_m,
                    mapping=p9.aes(x='ds',
                                   y='np.log(y+1)',
                                   color='unique_id')) + \
          p9.geom_line() + \
          PLOT_THEME + \
          p9.labs(x='', y='HMFD', color='Location') + \
          p9.theme(legend_position='right')

plot_uv.save('assets/outputs/plot_eda_uv_monthly.pdf', width=12, height=6)

df_uv_sub = df_uv[df_uv['ds'] < '2009-07-01']

plot_uv = p9.ggplot(data=df_uv_sub,
                    mapping=p9.aes(x='ds',
                                   y='np.log(y+1)',
                                   color='unique_id')) + \
          p9.geom_line() + \
          PLOT_THEME + \
          p9.labs(x='', y='HMFD', color='Location') + \
          p9.theme(legend_position='right')

plot_uv.save('assets/outputs/plot_eda_uv.pdf', width=12, height=6)
