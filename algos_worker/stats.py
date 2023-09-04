import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .algo import FederatedWorkerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class StatsFederatedWorkerAlgo(FederatedWorkerAlgo):
    def __init__(self, params: dict = {}):
        self.data: Optional[np.ndarray] = None        
        super().__init__(name="stats", params=params, model_suffix="json")

    def initialize(self):
        # self.params['cutoff'] = self.params.get('cutoff', 730)
        # self.params['delta'] = self.params.get('delta', 30)
        self.cutoff = self.params.get('cutoff')
        self.delta = self.params.get('delta')

        # will be used to store results during "training"
        self.results = {'logs': ''}

        logger.info("Reading data from %s.csv", self.key)
        # "training" will read from here (self.data)
        self.data = pd.read_csv(f"{self.key}.csv")
        logger.info("Data shape: %s", self.data.shape)
        logger.info("Initialized, ready for training")

    def survival_rate(self, df: pd.DataFrame, cutoff: int, delta: int) -> list:
        """ Compute survival rate at certain time points after diagnosis

        Parameters
        ----------
        df
            DataFrame with TNM data
        cutoff
            Maximum number of days for the survival rate profile
        delta
            Number of days between the time points in the profile

        Returns
        -------
        survival_rates
            Survival rate profile
        """

        # Get survival days, here we assume the date of last follow-up as death date
        df['date_of_diagnosis'] = pd.to_datetime(df['date_of_diagnosis'])
        df['date_of_fu'] = pd.to_datetime(df['date_of_fu'])
        df['survival_days'] = df.apply(
            lambda x: (x['date_of_fu'] - x['date_of_diagnosis']).days, axis=1
        )

        # Get survival rate after a certain number of days
        times = list(range(0, cutoff, delta))
        all_alive = len(df[df['vital_status'] == 'alive'])
        all_dead = len(df[df['vital_status'] == 'dead'])
        survival_rates = []
        for time in times:
            dead = len(
                df[(df['survival_days'] <= time) & (df['vital_status'] == 'dead')]
            )
            alive = (all_dead - dead) + all_alive
            survival_rates.append(alive / len(df))

        return survival_rates


    def train(self, callback=None):
        # statistics adapted from: https://github.com/MaastrichtU-CDS/v6-healthai-dashboard-py
        logger.info('Getting centre name')
        column = 'centre'
        if column in self.data.columns:
            centre = self.data[column].unique()[0]
            self.results['organisation'] = centre
        else:
            self.results['organisation'] = None
            self.results['logs'] += f'Column {column} not found in the data\n'

        logger.info('Counting number of unique ids')
        column = 'id'
        if column in self.data.columns:
            nids = self.data[column].nunique()
            self.results['nids'] = nids
        else:
            self.results['logs'] += f'Column {column} not found in the data\n'

        logger.info('Counting number of unique ids per stage')
        column = 'stage'
        if column in self.data.columns:
            self.data[column] = self.data[column].str.upper()
            stages = self.data.groupby([column])['id'].nunique().reset_index()
            self.results[column] = stages.to_dict()
        else:
            self.results['logs'] += f'Column {column} not found in the data'

        logger.info('Counting number of unique ids per vital status')
        column = 'vital_status'
        if column in self.data.columns:
            vital_status = self.data.groupby([column])['id'].nunique().reset_index()
            self.results[column] = vital_status.to_dict()
        else:
            self.results['logs'] += f'Column {column} not found in the data'

        logger.info('Getting survival rates')
        columns = ['date_of_diagnosis', 'date_of_fu']
        if (columns[0] in self.data.columns) and (columns[1] in self.data.columns):
            survival = self.survival_rate(self.data, self.cutoff, self.delta)
            self.results['survival'] = survival
        else:
            self.results['logs'] += \
                f'Columns {columns[0]} and/or {columns[1]} not found in the data'

        # Save results
        logger.info("Saving local statistics results")
        with open(self.model_path, "w+") as f:
            json.dump(self.results, f)
