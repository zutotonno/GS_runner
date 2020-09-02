import numpy as np
import json
import sqlalchemy
import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
import time
import models


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


class gs_runner:
    def __init__(self, json_path):
        with open(json_path) as json_buffer:
            json_data = json.load(json_buffer)
        # Runner informations
        self.settings = json_data['settings']
        self.data_info = json_data['data']
        self.connection_info = json_data['connection']
        # Grid Search
        self.model_grid = json_data['model']
        self.training_grid = json_data['training']
        # Database connection
        self.db_connection = None
        # Dataset attributes
        self.target_variable = None
        self.df_train = None
        self.df_test = None
        self.load_data()

    def create_connection(self):
        db_connection = sqlalchemy.\
            create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                          format(self.connection_info['database_username'],
                                 self.connection_info["database_password"],
                                 self.connection_info["database_ip"],
                                 self.connection_info["database_name"]))
        return db_connection

    def load_data(self):
        columns = self.data_info['columns']
        timestamp_variable = self.data_info['timestamp']
        self.target_variable = self.data_info['target']
        self.df_train = pd.read_sql_table(self.data_info['train_table'],
                                          con=self.db_connection,
                                          columns=columns).\
            set_index(timestamp_variable)
        self.df_test = pd.read_sql_table(self.data_info['test_table'],
                                         con=self.db_connection,
                                         columns=[timestamp_variable, self.target_variable]).\
            set_index(timestamp_variable)

    def iter_train(self):
        '''
        Take a dictionary of lists of training hyper parameters
        and return dictionries of training hyper parameters
        '''
        keys = self.training_grid.keys()
        values = self.training_grid.values()
        for upla in itertools.product(values):
            yield dict(zip(keys, upla))

    def iter_model(self):
        '''
        Take a dictionary of lists of model hyper parameters
        and return a list dictionaries of model hyper parameters
        '''
        keys = self.model_grid.keys()
        values = self.model_grid.values()
        for upla in itertools.product(values):
            yield dict(zip(keys, upla))

    def run(self):
        # Creating db connection
        self.db_connection = self.create_connection()
        # Loading data from database
        self.load_data()
        periods = len(self.df_test.index)
        # Start grid search
        for training_params in self.iter_train():
            for model_params in self.iter_model():
                mape = 0
                rmse = 0
                for _ in self.settings['n_exp']:
                    model = getattr(models, model_params['name'])(model_params)
                    model.train(training_params)

                    result = model.predict(periods)
                    mape += mean_absolute_percentage_error(
                        result, self.df_test)
                    rmse += mean_squared_error(result,
                                               self.df_test, squared=False)
                mape = mape / self.settings['n_exp']
                rmse = rmse / self.settings['n_exp']
        return


class gs_runner_old:

    def __init__(self, json_path, n_exp=10):
        with open(json_path) as f:
            self.gs_data = json.load(f)

        self.scaler = None
        self.df_TEST = None
        self.inital_seed = None

        self.target_name = self.gs_data['data']['target_name']
        self.input_names = self.gs_data['data']['input_names']
        self.n_exp = self.gs_data['model']["training"]['n_exp']

        database_connection = self.create_connection()
        self.df_TRAIN = pd.read_sql_table(self.gs_data['data']['table_train'],
                                          con=database_connection,
                                          columns=['timestamp'] + self.input_names + [self.target_name])
        self.df_TRAIN.set_index('timestamp', inplace=True)

        self.df_TEST = pd.read_sql_table(self.gs_data['data']['table_test'],
                                         con=database_connection,
                                         columns=['timestamp'] + self.input_names + [self.target_name])
        self.df_TEST.set_index('timestamp', inplace=True)

    def run(self):
        n_exp = self.gs_data['model']['training']['n_exp']
        params_values = [self.gs_data['model']['params'][i]
                         for i in self.gs_data['model']['params'].keys()]
        gs_over = list(itertools.product(*params_values))
        pbar = tqdm(total=len(gs_over))
        current_execution = 1
        for params_selected in gs_over:
            # model = GSmodel(json_path, gs_data.model_hparams)
            # model.train(self.df_train, gs_data.train_hparams)
            # output = model.forecast(num_timestamp)
            # error = Error_metric(output, self.df_test)
            # self.write_db(error, model_hparams, train_hparams)
            exp_id = int(round(time.time() * 1000))
            key_v = dict(
                zip(self.gs_data['model']['params'].keys(), params_selected))
            param_count = 0
            test_MAE = []
            test_MaxErr = []
            test_MAPE = []
            epochs = []
            models_list = []
            preds_list = []
            exp_ids = []
            train_time = 0
            train_X, train_y, init_seed = self.create_train_data(
                key_v['n_in'], key_v['n_out'])
            print('Dataset Train shape X, Y: ', train_X.shape, train_y.shape)
            for i in range(n_exp):
                print("****************************************\n")
                print("Execution: " + str(i + 1) + " of " + str(n_exp) +
                      " at " + str(current_execution) + " over: " + str(len(gs_over)))
                print('Using : ')
                print(params_selected)

                model_id = str(exp_id) + "_" + str(i)
                exp_ids.append(model_id)
                # Train phase
                model = self.create_model(**key_v)
                models_list.append(model)
                param_count = self.param_count(model)
                start = time.time()
                r = self.train_model(
                    model, train_X, train_y, key_v['batch_size'])
                end = time.time()
                r_train_time = end - start
                train_time += r_train_time
                epochs.append(r['epochs'])

                # Testing phase
                test_y = np.array(self.df_TEST)

                print('\nStart prediction on test ----------')
                initial_seed = np.array(init_seed)
                n_preds = int(test_y.shape[0] / key_v['n_out'])
                n_remaining = int(test_y.shape[0] % key_v['n_out'])
                inv_yhat = self.forecast(
                    model, initial_seed, n_steps=n_preds, n_remaining=n_remaining)
                preds_list.append(inv_yhat)

                test_MAE.append(mean_absolute_error(test_y, inv_yhat))
                test_MAPE.append(mean_absolute_percentage_error(
                    test_y + 1, inv_yhat + 1))
                test_MaxErr.append(max_error(test_y, inv_yhat))
                print(test_MAE, test_MAPE, test_MaxErr)

            train_time = train_time / n_exp
            ep_metrics = {'avg_train_time': train_time, 'min_epochs': np.min(
                epochs), 'avg_epochs': np.mean(epochs), 'max_epochs': np.max(epochs)}
            mae_metrics = {'min_MAE': np.min(test_MAE), 'avg_MAE': np.mean(
                test_MAE), 'max_MAE': np.max(test_MAE)}
            max_metrics = {'min_MAX': np.min(test_MaxErr), 'avg_MAX': np.mean(
                test_MaxErr), 'max_MAX': np.max(test_MaxErr)}
            mape_metrics = {'min_MAPE': np.min(test_MAPE), 'avg_MAPE': np.mean(
                test_MAPE), 'max_MAPE': np.max(test_MAPE)}

            data_metrics = self.gs_data['data']

            data_metrics['input_names'] = ''.join(data_metrics['input_names'])

            _model_args = [{'exp_id': exp_id},
                           {'model': self.gs_data['model']['name']},
                           key_v,
                           {'param_count': param_count},
                           self.gs_data['model']['training'],
                           ep_metrics,
                           mae_metrics,
                           max_metrics,
                           mape_metrics,
                           data_metrics]

            _columns = [[el for el in it.keys()] for it in _model_args]
            columns = [item for sublist in _columns for item in sublist]
            _data = [[str(el) for el in it.values()] for it in _model_args]
            data = [item for sublist in _data for item in sublist]

            df_to_db = pd.DataFrame(
                np.array(data).reshape(1, -1), columns=columns)
            df_to_db.to_sql(con=self.create_connection(
            ), name=self.gs_data['data']['table_result'], if_exists='append', index=None)
            for i in range(len(exp_ids)):
                model_id = exp_ids[i]
                inv_yhat = preds_list[i]
                col_name = str(i) + 'predicted'
                predictions = pd.DataFrame(pd.Series(data=inv_yhat.reshape(-1), index=self.df_TEST.index),
                                           columns=[col_name]).to_csv(path_or_buf=self.gs_data['data']['exp_folder'] + '/' + model_id + ".csv")
                self.save_model(models_list[i], name=model_id)
            current_execution += 1
            pbar.update(1)
        self.close()

    def create_connection(self):
        database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                                       format(self.gs_data["connection"]['database_username'],
                                                              self.gs_data["connection"]["database_password"],
                                                              self.gs_data["connection"]["database_ip"],
                                                              self.gs_data["connection"]["database_name"]))
        return database_connection

    def forecast(self, model, initial_seed: np.array, n_steps: int, n_remaining: int) -> np.array:
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
        pass

    def create_train_data(self, n_in: int, n_out: int) -> np.array:
        """[summary]
        Arguments:
            n_in {int} -- [num of input sample as input for the model]
            n_out {int} -- [num of model output]

        Returns:
            (train_X : np.array, train_y : np.array, inital_seed : pd.DataFrame)
        """
        pass

    def train_model(self, model, train_X: np.array, train_y: np.array, batch_size: int):
        """[summary]

        Arguments:
            model {[type]} -- [a class with the method 'predict(initial_seed)' to make predictions -
                                             i.e. : tensorflow.keras.models.Sequential()]
            train_X {np.array} -- [train input array]
            train_y {np.array} -- [train target array]
            batch_size {int} -- [description]
            current_execution {str} -- [id for the current training]

        Returns:
            ('epochs': int, 'train_loss':double)
        """
        pass

    def create_model(self, params) -> object:
        """[summary]

        Arguments:
            params {[dict]} -- [list of parameters needed for the model creation]

        Returns:
            model
        """
        pass

    def param_count(self, model) -> int:
        """[summary]

        Arguments:
            model {[type]} -- [model created]

        Returns:
            model trainable params
        """
        return 0

    def save_model(self, model, name):

        pass

    def close(self):

        pass
