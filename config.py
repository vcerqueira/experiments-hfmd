from pathlib import Path

import plotnine as p9

DATA_FILE_PATH = Path('assets/dataset.xlsx')
RESULTS_FILE_PATH = Path('assets/results')
TEST_SIZE = 365 * 2  # TWO YEARS
HORIZON = 14  # two weeks
N_LAGS = 14  # two weeks
PERIOD = 365  # 1 year ... assuming yearly seasonality
FREQ = 'D'  # daily granularity
META_COLUMNS = 'y|unique_id|ds|cutoff|horizon'

PLOT_THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
             p9.theme(plot_margin=.025,
                      panel_background=p9.element_rect(fill='white'),
                      plot_background=p9.element_rect(fill='white'),
                      legend_box_background=p9.element_rect(fill='white'),
                      strip_background=p9.element_rect(fill='white'),
                      legend_background=p9.element_rect(fill='white'),
                      axis_text_x=p9.element_text(size=9, angle=0),
                      axis_text_y=p9.element_text(size=9),
                      legend_title=p9.element_blank())

TWEEDIE_PARAMS = {'boosting_type': 'gbdt',
                  'lambda_l1': 0,
                  'lambda_l2': 1,
                  'learning_rate': 0.05,
                  'linear_tree': False,
                  'max_depth': 15,
                  'min_child_samples': 30,
                  'n_jobs': 1,
                  'num_leaves': 30,
                  'objective': 'tweedie'}

POISSON_PARAMS = {'boosting_type': 'gbdt',
                  'lambda_l1': 100,
                  'lambda_l2': 0,
                  'learning_rate': 0.05,
                  'linear_tree': False,
                  'max_depth': 5,
                  'min_child_samples': 15,
                  'n_jobs': 1,
                  'num_leaves': 30,
                  'objective': 'poisson'}

REGRESSION_PARAMS = {'boosting_type': 'gbdt',
                     'lambda_l1': 1,
                     'lambda_l2': 100,
                     'learning_rate': 0.05,
                     'linear_tree': False,
                     'max_depth': 5,
                     'min_child_samples': 15,
                     'n_jobs': 1,
                     'num_leaves': 3,
                     'objective': 'regression'}
