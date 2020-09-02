from BaseModel import GSModel as BaseModel
from sklearn.linear_model import LinearRegression

import numpy as np


class GSModel(BaseModel):
    def __init__(self, **kwargs):
        self.fit_intercept = kwargs['fit_intercept']
        self.target = kwargs['target']
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.time_span = None

    def train(self, df_train, **kwargs):
        self.time_span = np.arange(len(df_train))
        self.model.fit(self.time_span.reshape(1, -1),
                       df_train.loc[:, [self.target]].values)

    def forecast(self, num_timestamp):
        pred_time_span = 1 + self.time_span[-1] + np.arange(num_timestamp)
        return self.model.predict(pred_time_span)
