import warnings

import numpy as np
import pandas as pd

from config import DATA_FILE_PATH

warnings.filterwarnings("ignore")


class DataReader:
    UID_LIST = ['JOHOR', 'MELAKA', 'N9', 'KEDAH',
                'PERAK', 'PERLIS', 'PP', 'SELANGOR', 'WP',
                'KELANTAN', 'PAHANG', 'TERENGGANU',
                'SABAH', 'SARAWAK']

    UID_ZONE = {
        'JOHOR': 'Southern',
        'MELAKA': 'Southern',
        'N9': 'Southern',
        'KEDAH': 'Northern',
        'PERAK': 'Northern',
        'PERLIS': 'Northern',
        'PP': 'Northern',
        'SELANGOR': 'Central',
        'WP': 'Central',
        'KELANTAN': 'East Coast',
        'PAHANG': 'East Coast',
        'TERENGGANU': 'East Coast',
        'SABAH': 'Borneo',
        'SARAWAK': 'Borneo'
    }

    LOCATIONS_CODE = {
        'Southern': '_s',
        'Northern': '_n',
        'Central': '_c',
        'East Coast': '_ec',
        'Borneo': '_b',
    }

    EXOGENEOUS_VARS = ['temp', 'rain', 'rh']

    @classmethod
    def load_data(cls, drop_exogenous: bool = False):
        df = pd.read_excel(DATA_FILE_PATH, sheet_name=1)

        dataset_list = []
        for uid in cls.UID_LIST:
            uid_zone_code = cls.LOCATIONS_CODE[cls.UID_ZONE[uid]]

            exogenous_names = [f'{c}{uid_zone_code}' for c in cls.EXOGENEOUS_VARS]

            uid_df = df[[uid] + exogenous_names]

            uid_df.columns = ['y', 'temperature', 'rainfall', 'humidity']
            uid_df.loc[:, 'unique_id'] = uid
            uid_df['ds'] = pd.date_range(start='2009-01-01', periods=uid_df.shape[0], freq='D')

            dataset_list.append(uid_df)

        dataset = pd.concat(dataset_list, axis=0).reset_index(drop=True)

        if drop_exogenous:
            dataset = dataset.drop(columns=['temperature', 'rainfall', 'humidity'])

        return dataset

    @staticmethod
    def map_forecasting_horizon_col(cv):
        cv_g = cv.groupby(['unique_id','cutoff'])

        horizon = []
        for g, df in cv_g:
            df = df.sort_values('ds')
            h = np.asarray(range(1, df.shape[0] + 1))
            hs = {
                'horizon': h,
                'ds': df['ds'].values,
                'unique_id': df['unique_id'].values,
                'cutoff': df['cutoff'].values,
            }
            hs = pd.DataFrame(hs)
            horizon.append(hs)

        horizon = pd.concat(horizon)

        cv = cv.merge(horizon, on=['unique_id', 'ds','cutoff'])

        return cv

    @staticmethod
    def train_test_split(df: pd.DataFrame, horizon: int):
        df_by_unq = df.groupby('unique_id')

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values('ds')

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df
