import numpy as np
import json
import sqlalchemy
import itertools
from keras_tqdm import TQDMCallback
from tqdm import tqdm
import time
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.gs_runner_LSTM import gs_runner_LSTM


class gs_runner_LSTM_onestep(gs_runner_LSTM):

    def __init__(self,json_path):
        super().__init__(json_path)
        self.scaler = StandardScaler()
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tensorflow.Session(config=config)




    def forecast(self, model, init_seed: np.array, n_steps: int , n_remaining: int) -> np.array:
        """[summary]
        Forecast values using model, and for a number of times specified by n_forecast

        Arguments:
            model {[type]} -- [a class with the method 'predict(initial_seed)' to make predictions -
                                             i.e. : tensorflow.keras.models.Sequential()]
            initial_seed {np.array} -- [the input array used to start the forecasting loop]

            n_steps {int} -- [number of predictions ]

            n_remaining {int} -- [number of remaining sample to predict]
        Returns:
            np.array -- [returns the forecasted time-series with a dimension specified by n_forecast]
        """
        initial_seed = self.scaler.transform(np.array(init_seed))
        n_in = model.input_shape[1]
        predictions = []
        for i in range(self.df_TEST.shape[0]):
            _pred = model.predict(initial_seed.reshape(1,n_in,1)).flatten()
            initial_seed = np.concatenate((initial_seed[1:].flatten(), _pred[:1]))
            predictions.append(self.scaler.inverse_transform(_pred.reshape(-1,1)[0]).flatten())
        return np.array(predictions).flatten().reshape(-1,1)







