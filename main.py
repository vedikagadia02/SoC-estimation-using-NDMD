import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models.define_nn_arch import KoopmanNetwork
from models.nn_approach_1 import trainAE
from models.nn_approach_1 import trainModel, testModel
from models.nn_approach_2 import trainModel_2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

def makeDataset(file_name):
    data = np.load(file_name)
    new_col = np.full((len(data[0]),1), 1)                      # size [179,1]
    temp_dataset = data[0]                                      # size [179,6]
    print(temp_dataset.shape)
    new_temp = np.concatenate((new_col, temp_dataset), axis=1)  # size [179,7]
    for i, cycle in enumerate(data[1:]):
        new_col = np.full((len(data[0]),1), i+1)
        new_cycle = np.concatenate((new_col, cycle), axis=1)
        nt = np.concatenate([new_temp,new_cycle])
        new_temp = nt

    # data shape [30072,7]
    dataset = pd.DataFrame(data=new_temp, columns = ['cycle', 'capacity', 'voltage_measured', 'current_measured', 'temperature_measured', 'voltage_load', 'current_load', 'time'])
    print(dataset)

    C = dataset['capacity'][0]
    soh = []
    for i in range(len(dataset)):
        soh.append([dataset['capacity'][i] / C])    
    soh = pd.DataFrame(data=soh, columns=['SoH'])
    attribs=['capacity', 'voltage_measured', 'current_measured',
            'temperature_measured', 'current_load', 'voltage_load', 'time']
    train_dataset = dataset[attribs]
    sc = MinMaxScaler(feature_range=(0,1))
    train_dataset = sc.fit_transform(train_dataset)
    print(train_dataset.shape)
    print(soh.shape)

    return dataset, train_dataset, soh

def loadData(file_name, ntrain, ntest):
    
    data = np.load(file_name)
    idx = np.arange(0, data.shape[0],1)
    endIdx = data.shape[1]
    np.random.shuffle(idx)
    
    init_training_data = data[idx[:ntrain],:,:-1]
    init_testing_data = data[idx[ntrain:ntrain+ntest],:,:-1]

    num_samples, num_features = init_training_data.shape[0], init_training_data.shape[1] * init_training_data.shape[2]
    flattened_data = init_training_data.reshape(num_samples, num_features)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(flattened_data)
    training_data_norm = scaler.transform(flattened_data)
    training_data = training_data_norm.reshape(num_samples, init_training_data.shape[1], init_training_data.shape[2])
    
    num_samples, num_features = init_testing_data.shape[0], init_testing_data.shape[1] * init_testing_data.shape[2]
    flattened_data = init_testing_data.reshape(num_samples, num_features)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(flattened_data)
    testing_data_norm = scaler.transform(flattened_data)
    testing_data = testing_data_norm.reshape(num_samples, init_testing_data.shape[1], init_testing_data.shape[2])

    training_input_data = np.concatenate([training_data[:, i, :] for i in range(0,endIdx-31,1)], axis=0)
    training_target_data = np.concatenate([training_data[:, i:i+30, :] for i in range(1, endIdx-30, 1)], axis=0)

    # training_input_data = np.concatenate([training_data_norm[i, :endIdx, :] for i in range(0,ntrain,1)], axis=0)
    # training_target_data = np.concatenate([training_data_norm[i, :endIdx+1, :] for i in range(0, ntrain, 1)], axis=0)
    
    testing_input_data = np.concatenate([testing_data[:, i, :] for i in range(0,endIdx-1,1)], axis=0)
    testing_target_data = np.concatenate([testing_data[:, i, :] for i in range(1, endIdx, 1)], axis=0)

    training_dataset = TensorDataset(torch.Tensor(training_input_data), torch.Tensor(training_target_data))
    training_loader = DataLoader(training_dataset, batch_size = 128, shuffle = True)

    testing_dataset = TensorDataset(torch.Tensor(testing_input_data), torch.Tensor(testing_target_data))
    testing_loader = DataLoader(testing_dataset, batch_size = 16, shuffle = False)

    return training_loader, testing_loader

def print_training_loader_shape(training_loader):
    num_batches = len(training_loader)

    # Shape of a single batch
    for batch in training_loader:
        batch_shape = batch[0].shape
        break

    print(f"Number of batches: {num_batches}")
    print(f"Shape of a single batch: {batch_shape}")

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(device))
    
    '''
    CHANGE VALUES HERE
    '''
    file_name = ".\soc_dataset_7.npy"
    ntrain = 126
    ntest = 42
    indim = 6
    obsdim = 20
    s = 10      # number of steps to multiply kMatrix over
    a1, a2, a3, a4 = 1.0, 50.0, 10.0, 1e-6
    encoder_lr = 0.0005
    kMatrix_lr = 0.001
    weight_decay = 1e-8
    gamma = 0.995
    epochs = 30
    epochs_encoder = 5
    values_to_plot = []
    '''
    END OF CHANGE VALUES
    '''

    dataset, train_dataset, soh = makeDataset(file_name)
    training_loader, testing_loader = loadData(file_name, ntrain, ntest)
    print_training_loader_shape(training_loader)

    model = KoopmanNetwork(indim, obsdim).to(device)
    optimizer = torch.optim.Adam([
        {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 'lr': encoder_lr},
        {'params': [model.kMatrixDiag, model.kMatrixUT], 'lr': kMatrix_lr}
    ], 
    weight_decay=weight_decay
    )
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(1, epochs):  
        if(epoch <=epochs_encoder):
            loss_total = trainAE(epoch, device, training_loader, s, model, a1, a2, a3, a4, optimizer)
            print('Epoch {:d}: Training Loss {:.05f}'.format(epoch, loss_total))
        else:    
            loss_rec, loss_total, koopman = trainModel(epoch, device, training_loader, s, model, a1, a2, a3, a4, optimizer)
            print('Epoch {:d}: Training Loss {:.05f}'.format(epoch, loss_total))
            values_to_plot.append(loss_total)
            
            scheduler.step()
            for row in koopman:
                    formatted_row = [format(element, '.5f') for element in row]
                    print(' '.join(formatted_row))

    with torch.no_grad():
        test_loss, predicted_dataset, target_dataset, soh_pred, soh_tar = testModel(model, device, testing_loader)
        print('Epoch {:d}: Testing Loss {:.02f}'.format(epoch, test_loss))

    plt.plot(range(len(values_to_plot)), values_to_plot)
    plt.xlabel('Epoch Index')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()