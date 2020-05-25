import numpy as np
import json
import sqlalchemy
import itertools
from keras_tqdm import TQDMCallback
from tqdm import tqdm
import time
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.gs_runner_LSTM import gs_runner_LSTM


# convert series to supervised learning
def series_to_supervised_nan(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        if(i == -1):
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        else :
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.drop(df.head(n_in).index,inplace=True)
    return agg

class gs_runner_DENSE(gs_runner_LSTM):

    def __init__(self,json_path):
        super().__init__(json_path)
        self.scaler = StandardScaler()


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
        n_out = model.output_shape[1]
        n_in = model.input_shape[1]
        predictions = []
        for i in range(n_steps):
            _pred = model.predict(initial_seed.reshape(1,n_in)).flatten()
            initial_seed = np.concatenate((initial_seed[n_out:].flatten(), _pred[:n_in]))
            predictions.append(self.scaler.inverse_transform(_pred.reshape(-1,1)).flatten())
        if n_remaining > 0:
            _pred = model.predict(initial_seed.reshape(1,n_in)).flatten()
            predictions.append(self.scaler.inverse_transform(_pred[:n_remaining].reshape(-1,1)).flatten())
            return np.concatenate(np.array(predictions).flatten()).reshape(-1,1)
        else:
            return np.array(predictions).flatten().reshape(-1,1)


    def create_train_data(self, n_in: int, n_out:int) -> np.array:
        """[summary]
        Arguments:
            n_in {int} -- [num of input sample as input for the model]
            n_out {int} -- [num of model output]

        Returns:
            (train_X : np.array, train_y : np.array, inital_seed : pd.DataFrame)
        """



        sc_tr = self.scaler.fit(self.df_TRAIN[[self.nameFT]].values.reshape(-1,1))
        values_train = sc_tr.transform(self.df_TRAIN[[self.nameFT]].values.astype(dtype=float) )



        groups = list(range(0,values_train.shape[1]))
        n_features = len(groups)
        reframed_dataset = series_to_supervised_nan(values_train, n_in, n_out)
        reframed_columns = list(reframed_dataset.columns)
        target_var_label = 'var' + str(n_features)
        columns_to_drop = []
        for col_name in list(reframed_dataset.columns):
            is_forecasted_input = not (
                target_var_label in col_name or '-' in col_name)
            if is_forecasted_input:
                columns_to_drop.append(col_name)

        reframed_dataset.drop(columns_to_drop,axis=1,inplace=True)
        reframed_dataset.dropna(inplace=True)
        # split into train and validation sets
        values = reframed_dataset.values
        train = values
        n_features = 1
        n_obs = n_in*n_features # univariate

        train_X, train_y = train[:, :n_obs], train[:, -n_out:]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape(train_X.shape[0], n_in)
        print('Dataset Train shape X, Y: ',train_X.shape, train_y.shape)
        initial_seed = self.df_TRAIN[[self.nameFT]].iloc[-n_in:]
        return train_X, train_y, initial_seed


    def create_model(self,n_in=168, n_out=168,num_hidden=50, num_layers=1,dropout=0,batch_size=32, activation='tanh'):
        model = Sequential()
        model.add(Dense(num_hidden,activation=activation, input_shape=(n_in,)))
        if dropout>0:
                model.add(Dropout(dropout))
        for i in range(1,num_layers):
            model.add(Dense( num_hidden,activation=activation))
            if dropout>0:
                model.add(Dropout(dropout))
        model.add(Dense(n_out, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        print(model.summary())
        return model






