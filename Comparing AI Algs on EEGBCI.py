import os
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import numpy as np
import sklearn
import scalar
import layer

def run():
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    num_subjects = 10
    runs = [5, 9, 13]  # motor imagery: hands vs feet

    subjects_ica = {}

    # return ICA array for 10 subjects
    for i in range(num_subjects):
        subject = i
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)
        montage = make_standard_montage('standard_1005') #arrangemnent of electrodes
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))

        raw.crop(tmax=60.)
        raw.filter(14., 30.)

        picks = pick_types(raw.info, eeg=True)

        ica = ICA(n_components=15, random_state=97)
        ica.fit(raw)

        raw.load_data()
        #ica.plot_sources(raw, show_scrollbars=False)
        #ica.plot_properties(raw, picks=[0, 1])
        #ica.plot_scores()

        icaArray = ica.get_components()
        print(icaArray)
        subjects_ica[i] = icaArray
        #features = ica.get_sources()

    # split into training and testing set
    X_raw_train = []
    Y_raw_train = []
    X_raw_test = []
    Y_raw_test = []

    # set size of training and testing set
    num_train = 7
    num_test = num_subjects - num_train

    for i in range(num_train):
        X_raw_train.append(subjects_ica[i])
        Y_raw_train.append(i)

    for i in range(num_test):
        X_raw_test.append(subjects_ica[i])
        Y_raw_test.append(i)

    num_X_rows = len(icaArray)
    num_X_cols = len(icaArray[0])
    num_Y = len(subjects_ica)

    #standardize the data
    X_scalers = [scalar.get_scaler(X_raw_train[row,:]) for row in range(num_X_rows)]
    X_train = np.array([scalar.standardize(X_raw_train[row,:], X_scalers[row]) for row in range(num_X_rows)])

    Y_scalers = [scalar.get_scaler(Y_raw_train[row,:]) for row in range(num_Y)]
    Y_train = np.array([scalar.standardize(Y_raw_train[row,:], Y_scalers[row]) for row in range(num_Y)])

    X_test = np.array([scalar.standardize(X_raw_test[row,:], X_scalers[row]) for row in range(num_X_rows)])
    Y_test = np.array([scalar.standardize(Y_raw_test[row,:], Y_scalers[row]) for row in range(num_Y)])

    # check that standardization worked
    print([X_train[row,:].mean() for row in range(num_X_rows)]) # should be close to zero
    print([X_train[row,:].std() for row in range(num_X_rows)])  # should be close to one
    print([Y_train[row,:].mean() for row in range(num_Y)]) # should be close to zero
    print([Y_train[row,:].std() for row in range(num_Y)])  # should be close to one

    layer.init()

    # set dimensions of layers
    layers_dim = [num_X_rows, 4, 4, num_Y] # input layer --- hidden layers --- output layers
    neural_net = []

    # construct the network
    for layer_index in range(len(layers_dim)):
        if layer_index == 0: # if input layer
            neural_net.append(layer(layer_index, False, 0, layers_dim[layer_index], 'irrelevant'))
        elif layer_index+1 == len(layers_dim): # if output layer
            neural_net.append(layer(layer_index, True, layers_dim[layer_index-1], layers_dim[layer_index], activation='linear'))
        else:
            neural_net.append(layer(layer_index, False, layers_dim[layer_index-1], layers_dim[layer_index], activation='relu'))

    # check for unwanted overfitting of data
    pred_n_param = sum([(layers_dim[layer_index]+1)*layers_dim[layer_index+1] for layer_index in range(len(layers_dim)-1)])
    act_n_param = sum([neural_net[layer_index].W.size + neural_net[layer_index].b.size for layer_index in range(1,len(layers_dim))])
    print(f'Predicted number of hyperparameters: {pred_n_param}')
    print(f'Actual number of hyperparameters: {act_n_param}')
    print(f'Number of data: {num_X_cols}')
    if act_n_param >= num_X_cols:
        raise Exception('It will overfit.')


    # forward propagation
    def activation(input_, act_func):
        if act_func == 'relu':
            return np.maximum(input_, np.zeros(input_.shape))
        elif act_func == 'linear':
            return input_
        else:
            raise Exception('Activation function is not defined.')

    def forward_prop(input_vec, layers_dim=layers_dim, neural_net=neural_net):
        neural_net[0].A = input_vec # Define A in input layer for for-loop convenience
        for layer_index in range(1,len(layers_dim)): # W,b,Z,A are undefined in input layer
            neural_net[layer_index].Z = np.add(np.dot(neural_net[layer_index].W, neural_net[layer_index-1].A), neural_net[layer_index].b)
            neural_net[layer_index].A = activation(neural_net[layer_index].Z, neural_net[layer_index].activation)
        return neural_net[layer_index].A

    # back propagation
    def get_loss(y, y_hat, metric='mse'):
        if metric == 'mse':
            individual_loss = 0.5 * (y_hat - y) ** 2
            return np.mean([np.linalg.norm(individual_loss[:,col], 2) for col in range(individual_loss.shape[1])])
        else:
            raise Exception('Loss metric is not defined.')

    def get_dZ_from_loss(y, y_hat, metric):
        if metric == 'mse':
            return y_hat - y
        else:
            raise Exception('Loss metric is not defined.')


    def get_dactivation(A, act_func):
        if act_func == 'relu':
            return np.maximum(np.sign(A), np.zeros(A.shape)) # 1 if backward input >0, 0 otherwise; then diaganolize
        elif act_func == 'linear':
            return np.ones(A.shape)
        else:
            raise Exception('Activation function is not defined.')

    def backward_prop(y, y_hat, metric='mse', layers_dim=layers_dim, neural_net=neural_net, num_train_datum=num_train_datum):
        for layer_index in range(len(layers_dim)-1,0,-1):
            if layer_index+1 == len(layers_dim): # if output layer
                dZ = get_dZ_from_loss(y, y_hat, metric)
            else:
                dZ = np.multiply(np.dot(neural_net[layer_index+1].W.T, dZ),
                                 get_dactivation(neural_net[layer_index].A, neural_net[layer_index].activation))
            dW = np.dot(dZ, neural_net[layer_index-1].A.T) / num_train_datum
            db = np.sum(dZ, axis=1, keepdims=True) / num_train_datum

    # optimizing the network
    learning_rate = 0.01
    max_epoch = 1000000

    for epoch in range(1,max_epoch+1):
        Y_hat_train = forward_prop(X_train) # update y_hat
        backward_prop(Y_train, Y_hat_train) # update (dW,db)


    for layer_index in range(1,len(layers_dim)):        # update (W,b)
        neural_net[layer_index].W = neural_net[layer_index].W - learning_rate * neural_net[layer_index].dW
        neural_net[layer_index].b = neural_net[layer_index].b - learning_rate * neural_net[layer_index].db

    if epoch % 100000 == 0:
        print(f'{get_loss(Y_train, Y_hat_train):.4f}')