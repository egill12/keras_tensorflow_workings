'''
author: Ed Gill
This file contains the neccessary modules for creating the training and testing files for the machine.
'''

# This file is a simple implementation of the 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

features_to_use = ["spot_v_HF", "spot_v_MF", "spot_v_LF", "HF_ema_diff",
                   "MF_ema_diff", "LF_ema_diff", "LDN", "NY", "Asia", "target"]

def create_train_test_file(data_file, data_size, test_split):
    '''
    This module will create the traingin and testing files to be used in the ML RNN model.
    :return: training and testing data fils.
    '''
    # How large should the training data be?
    if data_size > data_file.shape[0]:
        # Overwrite the data_length to be 90% f file, with remaining 10% as train
        data_size = int(data_file.shape[0]*0.9)
        # adding a buffer of 5 forward steps before we start trading on test data
        test_size = data_file.shape[0] - (data_size + 5)
    test_size = int(data_size*test_split)
    # training size is the first x data points
    train_original = data_file.iloc[:int(data_size), :].reset_index(drop= False)  # eurusd_train.iloc[-DATA_SIZE:,:]
    test_original = data_file.iloc[int(data_size) + 5: (int(data_size) + int(test_size)), :].reset_index(drop= False)
    return train_original , test_original


def create_dataset(dataset, populate_target, look_back, test):
    '''
    This creates the data for  passing to the LSTM module
    :param dataset:
    :param populate_target:
    :param look_back:
    :return:
    '''
    dataX, dataY, target_dates = [], [], []
    for i in range(len(dataset) - look_back + 1):
        # this takes the very last col as the target
        a = dataset[i:(i + look_back), :-1]
        dataX.append(a)
        # this code assumes that the target vector is the very last col.
        dataY.append(dataset[i + look_back - 1, -1])
        if populate_target:
            target_dates.append(test['Date'].loc[i + look_back - 1])
    return np.array(dataX), np.array(dataY), target_dates

def signal(output, thold):
    '''
    :param x: Create a signal from the predicted softmax activation output
    :return: signal to trade 
    '''
    if output >= thold:
        return 1
    elif output <= (1-thold):
        return -1
    else:
        return 0

def get_accuracy(predicted, test_target ):
    '''
    :return: the prediction accuracy of our model
    '''
    true_class = [np.sign(i[0]) for i in test_target]
    return accuracy_score(true_class, predicted)

def get_scaled_returns():
    '''
    This file will scale exposure based on the next 24 hour ahead prediction
    :return: 
    '''
    pass

def main():
    pass

if __name__ == "__main__":
    main()
