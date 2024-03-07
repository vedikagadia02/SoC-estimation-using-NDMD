import torch
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from ..models.define_nn_arch import KoopmanNetwork
from ..models.nn_approach_1 import trainAE
from ..models.nn_approach_1 import trainModel, testModel
from ..models.nn_approach_2 import trainModel_2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..plot_functions.overall_plots import plot_soc, plot_loss

### FILES IN FOLLOWING FILE STRUCTURE .\Desktop\SOC-estimation-using-NDMD\stock_implementation
### RUN WITH THIS COMMAND FROM DESKTOP (PARENT DIRECTORY OF SOC_IMPLEMENTATION_NDMD) python -m SoC-estimation-using-NDMD.stock_implementation.test

def loadData(train_file_name, test_files, test_dir_name, train_batch_size, test_batch_size):

    training_dataframe = pd.read_csv(train_file_name)
    training_data = training_dataframe.values
    training_input_data = training_data[:-30]
    training_target_data = np.array([training_data[i:i+30] for i in range(training_data.shape[0]-30)])
    training_dataset = TensorDataset(torch.Tensor(training_input_data), torch.Tensor(training_target_data))
    training_loader = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
    
    testing_data = {}
    testing_input_data = {}
    testing_target_data = {}
    testing_dataset = {}
    testing_loader = {}
    for file in test_files:
        dataframe = pd.read_csv(test_dir_name + file)
        data = dataframe.values
        key = file.replace('.csv','')
        input_data = data[:-30]
        target_data = np.array([data[i:i+30] for i in range(data.shape[0]-30)])
        dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(target_data))
        tensor_loader = DataLoader(dataset, batch_size = test_batch_size, shuffle = False)
        testing_data[key] = data
        testing_input_data[key] = input_data
        testing_target_data[key] = target_data
        testing_dataset[key] = dataset
        testing_loader[key] = tensor_loader

    return training_loader, testing_loader

def get_time_avg(predicted_values):
    sequence_size = predicted_values.shape[1]
    soc_predicted = np.zeros(predicted_values.shape[0])
    
    for i in range(predicted_values.shape[0]):
        soc_predicted[i] = predicted_values[i,-1]
    return soc_predicted

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(device))

    '''
    CHANGE VALUES HERE
    '''
    exp = 'soc'
    project_dir = './SOC-estimation-using-NDMD/stock_implementation/'
    test_dir_name = project_dir + "dataset/"
    train_file_name = project_dir + "dataset/train_dataset.csv"
    model_dir = project_dir + "checkpoints"
    koopman_dir = project_dir + "KoopmanPerEpoch"
    loss_dir = project_dir + "loss"
    plot_dir = project_dir + "plots/test"
    test_files = ['test_dataset.csv']

    train_batch_size = 128
    test_batch_size = 16
    indim = 6
    obsdim = 30
    s = 30 
    a1, a2, a3, a4 = 1.0, 50.0, 10.0, 1e-6
    encoder_lr = 0.001
    kMatrix_lr = 0.01
    weight_decay = 1e-7
    gamma = 0.995
    epochs = 50
    epochs_encoder = 5
    '''
    END OF CHANGE VALUES
    '''

    training_loader, testing_loader = loadData(train_file_name, test_files, test_dir_name, train_batch_size, test_batch_size)    
    model = KoopmanNetwork(indim, obsdim).to(device)
    model.load_state_dict(torch.load(model_dir+'/model_epoch_10.pt')) 

    epoch = -1
    ### here test_idx is used to differentiate between temperature-wise SOC test datasets
    for key in testing_loader:
        test_loader = testing_loader[key]    
        with torch.no_grad():
            test_loss, actual_values, predicted_values = testModel(exp, key, epoch, device, model, test_loader, plot_dir)
            soc_actual = actual_values
            soc_predicted = get_time_avg(predicted_values)
            mse_loss = np.mean((soc_actual - soc_predicted)**2)
            rmse = np.sqrt(mse_loss)
            print(f'Epoch {epoch}: Testing Loss {test_loss} for {key}')
            print(f'SOC RMSE error for {key} : {rmse}')
            plot_soc(plot_dir, key, epoch, soc_actual[30:], soc_predicted[:-30])