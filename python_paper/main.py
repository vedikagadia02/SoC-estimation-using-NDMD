import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models.define_nn_arch import KoopmanNetwork
from models.nn_working import trainModel
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
    # training_input_data = np.concatenate([training_data[:, i, :] for i in range(0,endIdx-1,1)], axis=0)
    # training_target_data = np.concatenate([training_data[:, i, :] for i in range(1, endIdx, 1)], axis=0)

    training_input_data = np.concatenate([training_data_norm[i, :endIdx, :] for i in range(0,ntrain,1)], axis=0)
    training_target_data = np.concatenate([training_data_norm[i, :endIdx+1, :] for i in range(0, ntrain, 1)], axis=0)
    
    testing_input_data = np.concatenate([testing_data[:, i, :] for i in range(0,endIdx-1,1)], axis=0)
    testing_target_data = np.concatenate([testing_data[:, i, :] for i in range(1, endIdx, 1)], axis=0)

    training_dataset = TensorDataset(torch.Tensor(training_input_data), torch.Tensor(training_target_data))
    training_loader = DataLoader(training_dataset, batch_size = 16, shuffle = False)

    testing_dataset = TensorDataset(torch.Tensor(testing_input_data), torch.Tensor(testing_target_data))
    testing_loader = DataLoader(testing_dataset, batch_size = 16, shuffle = False)

    return training_loader, testing_loader

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(device))
    scheduler = None
    ntrain = 300
    ntest = 100
    training_loader, testing_loader = loadData('./equation.npy', ntrain, ntest)

    model = KoopmanNetwork(2, 3).to(device)

    parameters = [{'params': [model.kMatrix]},
                  {'params': [model.encoder.parameters(), model.decoder.parameters()]}]
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
    # scheduler = ExponentialLR(optimizer, gamma = 0.995)

    for epoch in range(1, 10):  
        loss = trainModel(device, training_loader, model, optimizer)
        print('Epoch {:d}: Training Loss {:.05f}'.format(epoch, loss))
        # scheduler.step()
        koopman = model.getKoopmanMatrix()
        for row in koopman:
            formatted_row = [format(element, '.5f') for element in row]
            print(' '.join(formatted_row))