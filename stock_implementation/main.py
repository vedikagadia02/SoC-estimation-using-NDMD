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
from ..plot_functions.overall_plots import plot_stock, plot_loss

### FILES IN FOLLOWING FILE STRUCTURE .\Desktop\SOC-estimation-using-NDMD\stock_implementation
### RUN WITH THIS COMMAND FROM DESKTOP (PARENT DIRECTORY OF SOC_IMPLEMENTATION_NDMD) python -m SoC-estimation-using-NDMD.stock_implementation.main

def data_scaler(data):
    num_samples, num_features = data.shape[0], data.shape[1]
    flattened_data = data.reshape(num_samples, num_features)
    scaler = MinMaxScaler(feature_range=(0,1))
    # scaler = StandardScaler()
    scaler.fit(flattened_data)
    data_norm = scaler.transform(flattened_data)
    scaled_data = data_norm.reshape(num_samples, num_features)
    return scaled_data, scaler

def loadData(train_files, test_files, dataset_dir_name, past, s, train_batch_size, test_batch_size):

    training_data = {}
    training_scaler = {}
    training_input_data = {}
    training_target_data = {}
    training_dataset = {}
    training_loader = {}
    for file in train_files:
        dataframe = pd.read_csv(dataset_dir_name + file)
        init_data = dataframe.values
        data, scaler = data_scaler(init_data)
        endIdx = data.shape[0]
        key = file.replace('.csv','')
        input_data = np.concatenate([data[:, i:i+past, :] for i in range(0, endIdx-past-s, s)], axis=0)
        target_data = np.concatenate([data[:, i:i+s, :] for i in range(past-1, endIdx-s+1, past+s)], axis=0)
        input_data = data[:-1*s]
        target_data = np.array([data[i:i+s] for i in range(data.shape[0]-s)])
        dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(target_data))
        tensor_loader = DataLoader(dataset, batch_size = train_batch_size, shuffle = False)
        training_data[key] = data
        training_scaler[key] = scaler
        training_input_data[key] = input_data
        training_target_data[key] = target_data
        training_dataset[key] = dataset
        training_loader[key] = tensor_loader

    testing_data = {}
    testing_scaler = {}
    testing_input_data = {}
    testing_target_data = {}
    testing_dataset = {}
    testing_loader = {}
    for file in test_files:
        dataframe = pd.read_csv(dataset_dir_name + file)
        init_data = dataframe.values
        data, scaler = data_scaler(init_data)
        key = file.replace('.csv','')
        input_data = data[:-1*s]
        target_data = np.array([data[i:i+s] for i in range(data.shape[0]-s)])
        dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(target_data))
        tensor_loader = DataLoader(dataset, batch_size = test_batch_size, shuffle = False)
        testing_data[key] = data
        testing_scaler[key] = scaler
        testing_input_data[key] = input_data
        testing_target_data[key] = target_data
        testing_dataset[key] = dataset
        testing_loader[key] = tensor_loader

    return training_loader, testing_loader, training_scaler, testing_scaler

def print_training_loader_shape(train_loader):
    for key in train_loader:
        training_loader = train_loader[key]
        num_batches = len(training_loader)
        for batch in training_loader:
            batch_shape = batch[0].shape
            break

        print(f"{key[6:]} : Number of batches: {num_batches}")
        print(f"{key[6:]} : Shape of a single batch: {batch_shape}")

def make_dir(model_dir, koopman_dir, loss_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(koopman_dir):
        os.makedirs(koopman_dir, exist_ok=True)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir, exist_ok=True)

def get_time_avg(predicted_values):
    sequence_size = predicted_values.shape[1]
    stock_predicted = np.zeros(predicted_values.shape[0])
    
    for i in range(predicted_values.shape[0]):
        # values = []
        # for j in range(min(i+1, sequence_size)):
        #     values.append(predicted_values[i-j, j])
        # stock_predicted[i] = np.mean(values)
        stock_predicted[i] = predicted_values[i,-1]
    return stock_predicted

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(device))

    '''
    CHANGE VALUES HERE
    '''
    exp = 'stock'
    project_dir = './SOC-estimation-using-NDMD/stock_implementation/'
    dataset_dir_name = project_dir + "dataset/"
    # train_files = ['train_HDFCBANK.csv', 'train_RELIANCE.csv', 'train_DIVISLAB.csv']
    # test_files = ['test_HDFCBANK.csv', 'test_RELIANCE.csv', 'test_DIVISLAB.csv']
    train_files = ['train_TATAMOTORS.csv']
    test_files = ['test_TATAMOTORS.csv']
    model_dir = project_dir + "checkpoints"
    koopman_dir = project_dir + "KoopmanPerEpoch"
    loss_dir = project_dir + "loss"
    plot_dir = project_dir + "plots"
    
    train_batch_size = 128
    test_batch_size = 16
    indim = 12
    obsdim = 50
    past = 10
    s = 30
    a1, a2, a3, a4 = 1.0, 50.0, 10.0, 1e-6
    encoder_lr = 0.0001
    kMatrix_lr = 0.001
    weight_decay = 1e-7
    gamma = 0.99
    epochs = 56
    epochs_encoder = 5
    '''
    END OF CHANGE VALUES
    '''

    make_dir(model_dir, koopman_dir, loss_dir)
    training_loader, testing_loader, training_scaler, testing_scaler = loadData(train_files, test_files, dataset_dir_name, past, s, train_batch_size, test_batch_size)    
    print_training_loader_shape(training_loader)

    train_losses = {key: [] for key in training_loader.keys()}
    test_losses = {key: [] for key in testing_loader.keys()}

    '''
    initialise the model
    '''
    model = KoopmanNetwork(indim, obsdim).to(device)
    optimizer = torch.optim.Adam([
        {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 'lr': encoder_lr},
        {'params': [model.kMatrixDiag, model.kMatrixUT], 'lr': kMatrix_lr}
    ], 
    weight_decay=weight_decay
    )
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    '''
    train the model
    '''

    for key in testing_loader:
        ticker = key[4:]
        train_loader = training_loader['train'+ticker]
        train_scaler = training_scaler['train'+ticker]
        test_loader = testing_loader['test'+ticker]
        test_scaler = testing_scaler['test'+ticker]   
        loss_key_path = loss_dir + '/' + key[5:]

        if not os.path.exists(loss_key_path):
            os.makedirs(loss_key_path)

        for epoch in range(1, epochs):  
            start_time = time.time()

            ### autoencoder
            if(epoch <=epochs_encoder):
                loss_total = trainAE(epoch, device, train_loader, s, model, a1, a2, a3, a4, optimizer)
                print(f'{ticker[1:]} : Epoch {epoch}: Training Loss {loss_total}')
                loss_save_path = os.path.join(loss_key_path, 'train.txt')
                with open(loss_save_path, 'a') as f:
                    f.write('Epoch {:d}: Training Loss {:.05f}\n'.format(epoch, loss_total))

            ### koopman
            else:    
                loss_rec, loss_total, koopman = trainModel(epoch, device, train_loader, s, model, a1, a2, a3, a4, optimizer)
                print(f'{ticker[1:]} : Epoch {epoch}: Training Loss {loss_total}')
                train_losses['train'+ticker].append(loss_total)
                loss_save_path = os.path.join(loss_key_path, 'train.txt')
                with open(loss_save_path, 'a') as f:
                    f.write('Epoch {:d}: Training Loss {:.05f}\n'.format(epoch, loss_total))
                scheduler.step()
                koopman_save_path = os.path.join(koopman_dir, f"koopman_epoch_{epoch}.txt")
                with open(koopman_save_path, "w") as f:
                    for row in koopman:
                        formatted_row = [format(element, '.5f') for element in row]
                        f.write(' '.join(formatted_row) + "\n")
            
            '''
            save, test, plot the model
            '''
            if epoch % 5 == 0:
                model_save_path = os.path.join(model_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), model_save_path)

                # for key in testing_loader:
                 
                with torch.no_grad():
                    test_loss, mm_scale_actual, stock_predicted, iw_scale_actual, iw_scale_predicted = testModel(exp, key, epoch, device, model, test_loader, plot_dir, test_scaler)
                    test_losses[key].append(test_loss)
                    mm_scale_predicted = get_time_avg(stock_predicted)
                    mse_loss = np.mean((iw_scale_actual - iw_scale_predicted)**2)
                    rmse = np.sqrt(mse_loss)
                    print(f'{ticker[1:]} : Epoch {epoch}: Testing Loss {test_loss}')
                    print(f'{ticker[1:]} : RMSE error {rmse}')
                    loss_save_path = os.path.join(loss_key_path, 'test.txt')
                    with open(loss_save_path, 'a') as f:
                        f.write(f'Epoch {epoch} : Testing Loss {test_loss}\n')
                        f.write(f'Epoch {epoch} : RMSE Loss {rmse}\n')
                    plot_stock(plot_dir, key, epoch, mm_scale_actual, mm_scale_predicted, iw_scale_actual, iw_scale_predicted)

            end_time = time.time()  
            elapsed_time = end_time - start_time  
            print('Epoch {:d}: Elapsed time {:.02f} seconds'.format(epoch, elapsed_time))
        
    for key in test_losses:
        ticker = key[4:]
        plot_loss(plot_dir, key, train_losses['train'+ticker], test_losses[key])