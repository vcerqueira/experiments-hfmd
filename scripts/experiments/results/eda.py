import pandas as pd

from codebase.data_reader import DataReader

df = DataReader.load_data()

df = df.drop(['unique_id', 'ds'], axis=1)

corr_df = df.corr()
corr_df = df.corr(method='kendall')

# https://nixtlaverse.nixtla.io/statsforecast/index.html
# https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/intermittentdata.html
# ml local v ml global

# whta is the best approach
# can ML and DL leverage diff locations?


dfm = corr_df.reset_index().melt('index')
dfm['index'] = dfm['index'].apply(lambda x: x.capitalize()).replace({'Y': 'HMFD'})
dfm['variable'] = dfm['variable'].apply(lambda x: x.capitalize()).replace({'Y': 'HMFD'})
dfm.columns = ['v1', 'v2', 'Correlation']

import plotnine as p9

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

plot = \
    p9.ggplot(data=dfm,
              mapping=p9.aes(x='v1',
                             y='v2', fill='Correlation')) + \
    p9.geom_tile(p9.aes(width=0.95, height=0.95)) + \
    base_theme + \
    p9.theme(axis_title=p9.element_blank(),
             axis_text=p9.element_text(size=12)) + \
    p9.scale_fill_gradient2()

#
