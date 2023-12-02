import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models.define_nn_arch import KoopmanNetwork
from models.nn_approach_1 import trainAE
from models.nn_approach_1 import trainModel
from models.nn_approach_2 import trainModel_2
from sklearn.preprocessing import StandardScaler

def loadData(file_name, ntrain, ntest):
    data = np.load(file_name)
    idx = np.arange(0, data.shape[0],1)
    endIdx = data.shape[1]
    np.random.shuffle(idx)
    
    training_data = data[idx[:ntrain],:,:-1]
    testing_data = data[idx[ntrain:ntrain+ntest],:,:-1]

    num_samples, num_features = training_data.shape[0], training_data.shape[1] * training_data.shape[2]
    flattened_data = training_data.reshape(num_samples, num_features)
    scaler = StandardScaler()
    scaler.fit(flattened_data)
    training_data_norm = scaler.transform(flattened_data)
    training_data_norm = training_data_norm.reshape(num_samples, training_data.shape[1], training_data.shape[2])
    
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
    dataset = "./equation_2.npy"
    ntrain = 1200
    ntest = 400
    indim = 2
    obsdim = 4
    s = 30      # number of steps to multiply kMatrix over
    a1, a2, a3, a4 = 1.0, 50.0, 10.0, 1e-6
    encoder_lr = 0.0005
    kMatrix_lr = 0.001
    weight_decay = 1e-7
    gamma = 0.995
    epochs = 250
    epochs_encoder = 5
    '''
    END OF CHANGE VALUES
    '''

    training_loader, testing_loader = loadData(dataset, ntrain, ntest)
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
            scheduler.step()
            for row in koopman:
                    formatted_row = [format(element, '.5f') for element in row]
                    print(' '.join(formatted_row))