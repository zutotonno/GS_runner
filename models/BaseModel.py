import numpy as np


class GSModel():
    def __init__(self, json_path):
        self.json_path = json_path
        self.model = None

    def initialize(self, init_seed, **kwargs):
        '''
        Takes in input a list of hyper-parameter and intialize the model
        by populating the attribute "self.model".
        A seed init_seed has to be provided to replicate the model.
        N.B Hyper-Paramters related to the training (length_sequence,
        batch_size, lerning_rate, momentum, etc.) has not to be passed here.
        '''
        return None

    def train(self, training_dataset, train_seed, **kwargs):
        '''
        Takes in input a training dataset (pandas dataset with timestamp index
        or a numpy array) and a list of hyperparameter to train the model.
        A seed has to be provided to replicate the result.
        '''
        return None

    def forecast(self, num_timestamp):
        '''
        Make a prediction of lengh num_timestamp.
        '''
        prediction = None
        return prediction
