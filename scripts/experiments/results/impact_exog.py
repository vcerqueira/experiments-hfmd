import pandas as pd
import plotnine as p9

data_dict = {
    'LGBM-Univariate': 0.5782092504903217,
    'LGBM-Exogenous': 0.5782932131956504,
    'NHITS-Univariate': 0.5610176578032522,
    'NHITS-Exogenous': 0.5634947914768041,
}

# Convert dict to DataFrame and split the keys into model and features
df_rows = []
for key, value in data_dict.items():
    model, features = key.split('-')
    df_rows.append({
        'Model': model,
        'Approach': features,
        'SMAPE': value
    })

df = pd.DataFrame(df_rows)

print(df.to_latex(label='tab:exog', caption='CAPTION'))

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

plot = p9.ggplot(df, mapping=p9.aes(x='Model',
                                    y='SMAPE', fill='Approach')) + \
       p9.geom_bar(stat='identity',
                   position=p9.position_dodge(width=0.9), width=0.8) + \
       p9.scale_fill_brewer(type='qual', palette='Set2') + \
       p9.labs(title='',
               x='',
               y='SMAPE') + \
       base_theme + \
       p9.theme(
           axis_text_x=p9.element_text(angle=0, size=12),
           plot_title=p9.element_text(hjust=0.5)
       )

plot.save('assets/outputs/plot_exog.pdf', width=11, height=7)
