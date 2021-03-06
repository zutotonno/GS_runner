import numpy as np
import json
import sqlalchemy
import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_error,max_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

class gs_runner:

    def __init__(self, json_path):
        with open(json_path) as f:
            self.gs_data = json.load(f)
        if self.gs_data['model']['training']['scaler_mode'] == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.df_TEST = None
        self.inital_seed = None


        self.target_name = self.gs_data['data']['target_name']
        self.input_names = self.gs_data['data']['input_names']
        self.n_exp = self.gs_data['model']["training"]['n_exp']

        database_connection = self.create_connection()
        self.df_TRAIN = pd.read_sql_table(self.gs_data['data']['table_train'],
                           con=database_connection, columns=['timestamp']+self.input_names+[self.target_name])
        self.df_TRAIN.set_index('timestamp',inplace=True)

        self.df_TEST = pd.read_sql_table(self.gs_data['data']['table_test'],
                           con=database_connection, columns=['timestamp']+self.input_names+[self.target_name])
        self.df_TEST.set_index('timestamp',inplace=True)





    def run(self):
        n_exp = self.gs_data['model']['training']['n_exp']
        params_values = [self.gs_data['model']['params'][i] for i in self.gs_data['model']['params'].keys()]
        gs_over = list(itertools.product(*params_values))
        pbar = tqdm(total=len(gs_over))
        current_execution = 1
        for params_selected in gs_over:
            exp_id = int(round(time.time() * 1000))
            key_v = dict(zip(self.gs_data['model']['params'].keys(), params_selected))
            param_count = 0
            test_MAE = []
            test_MaxErr = []
            test_MAPE = []
            test_RMSE = []
            epochs = []
            models_list = []
            preds_list = []
            exp_ids = []
            train_time = 0
            train_X,train_y,init_seed = self.create_train_data(key_v['n_in'], key_v['n_out'])
            print('Dataset Train shape X, Y: ',train_X.shape, train_y.shape)
            for i in range(n_exp):
                print("****************************************\n")
                print("Execution: " + str(i+1) + " of "+str(n_exp)+" at "+  str(current_execution) + " over: " + str(len(gs_over)))
                print('Using : ')
                print(params_selected)

                model_id = str(exp_id)+"_"+str(i)
                exp_ids.append(model_id)
                # Train phase
                model = self.create_model(**key_v)
                models_list.append(model)
                param_count = self.param_count(model)
                start = time.time()
                r = self.train_model(model, train_X,train_y, key_v['batch_size'])
                end = time.time()
                r_train_time = end-start
                train_time += r_train_time
                epochs.append(r['epochs'])


                ### Testing phase
                test_y = np.array(self.df_TEST)

                print('\nStart prediction on test ----------')
                initial_seed = np.array(init_seed)
                n_preds = int(test_y.shape[0] / key_v['n_out'])
                n_remaining = int(test_y.shape[0] % key_v['n_out'])
                inv_yhat = self.forecast(model, initial_seed, n_steps=n_preds, n_remaining=n_remaining)
                preds_list.append(inv_yhat)


                test_MAE.append(mean_absolute_error(test_y, inv_yhat))
                test_MAPE.append(mean_absolute_percentage_error(test_y+1, inv_yhat+1))
                test_MaxErr.append(max_error(test_y, inv_yhat))
                test_RMSE.append(mean_squared_error(test_y, inv_yhat, squared=False))
                print(test_MAE, test_MAPE, test_MaxErr, test_RMSE)

            train_time = train_time/n_exp
            ep_metrics = {'avg_train_time':train_time,'min_epochs':np.min(epochs), 'avg_epochs':np.mean(epochs), 'max_epochs':np.max(epochs)}
            mae_metrics = {'min_MAE':np.min(test_MAE), 'avg_MAE':np.mean(test_MAE), 'max_MAE':np.max(test_MAE)}
            max_metrics = {'min_MAX':np.min(test_MaxErr), 'avg_MAX':np.mean(test_MaxErr), 'max_MAX':np.max(test_MaxErr)}
            mape_metrics = {'min_MAPE':np.min(test_MAPE), 'avg_MAPE':np.mean(test_MAPE), 'max_MAPE':np.max(test_MAPE)}
            rmse_metrics = {'min_MAPE':np.min(test_RMSE), 'avg_MAPE':np.mean(test_RMSE), 'max_MAPE':np.max(test_RMSE)}

            data_metrics = self.gs_data['data']

            data_metrics['input_names'] = ''.join(data_metrics['input_names'])

            _model_args = [{'exp_id': exp_id},
            {'model':self.gs_data['model']['name']}, key_v, {'param_count': param_count},self.gs_data['model']['training'], ep_metrics, mae_metrics, max_metrics, mape_metrics,rmse_metrics, data_metrics]

            _columns = [[el for el in it.keys()] for it in _model_args]
            columns = [item for sublist in _columns for item in sublist]
            _data = [[str(el) for el in it.values()] for it in _model_args]
            data = [item for sublist in _data for item in sublist]

            df_to_db = pd.DataFrame(np.array(data).reshape(1,-1), columns=columns)
            df_to_db.to_sql(con=self.create_connection(), name=self.gs_data['data']['table_result'], if_exists='append', index=None)
            for i in range(len(exp_ids)):
                model_id = exp_ids[i]
                inv_yhat = preds_list[i]
                col_name = str(i)+'predicted'
                predictions = pd.DataFrame(pd.Series(data=inv_yhat.reshape(-1), index=self.df_TEST.index),
                                            columns=[col_name]).to_csv(path_or_buf=self.gs_data['data']['exp_folder']+'/'+model_id+".csv")
                self.save_model(models_list[i],name=model_id)
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

    def forecast(self, model, initial_seed:np.array,  n_steps: int , n_remaining: int) -> np.array:
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


    def create_train_data(self,n_in: int, n_out:int) -> np.array:
        """[summary]
        Arguments:
            n_in {int} -- [num of input sample as input for the model]
            n_out {int} -- [num of model output]

        Returns:
            (train_X : np.array, train_y : np.array, inital_seed : pd.DataFrame)
        """
        pass


    def train_model(self, model,train_X : np.array, train_y : np.array,  batch_size: int):
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







