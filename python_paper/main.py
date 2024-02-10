import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models.define_nn_arch import KoopmanNetwork
from models.nn_approach_1 import trainAE,testModel
from models.nn_approach_1 import trainModel
from models.nn_approach_2 import trainModel_2
from sklearn.preprocessing import StandardScaler
import os

import matplotlib.pyplot as plt
import numpy as np

def save_actual_vs_predicted_plots(actual_values, predicted_values, epoch, save_folder="plots"):
    # actual_values [2000,2]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.figure(figsize=(10, 6))
    plt.plot(actual_values, label='actual', color='red')
    plt.plot(predicted_values, label='predicted', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted (Epoch {epoch})')
    plt.savefig(os.path.join(save_folder, f'plot_epoch_{epoch}.png'))  # Save the plot
    plt.close()  # Close the plot to release memory

# Define directory paths to save models and Koopman matrices
model_save_dir = "./checkpoints"
koopman_save_dir = "./KoopmanPerEpoch"

# Create directories if they don't exist
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(koopman_save_dir, exist_ok=True)

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
    dataset = "./dataset/equation_1/equation_1.npy"
    ntrain = 1200
    ntest = 400
    indim = 2
    obsdim = 3
    s = 30      # number of steps to multiply kMatrix over
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
    test_losses = []
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

            #Saving the Koopman matrix after every epoch
            koopman_save_path = os.path.join(koopman_save_dir, f"koopman_epoch_{epoch}.txt")
            with open(koopman_save_path, "w") as f:
                for row in koopman:
                    formatted_row = [format(element, '.5f') for element in row]
                    f.write(' '.join(formatted_row) + "\n")
            
            #Testing the model after every epoch and saving the point
            with torch.no_grad():
                test_loss,actual_values,predicted_values = testModel(device, model,testing_loader)
                test_losses.append(test_loss)
                save_actual_vs_predicted_plots(actual_values,predicted_values,epoch)
                print('Epoch {:d}: Testing Loss {:.02f}'.format(epoch, test_loss))

        
        
        #Saving the model at every 5th epoch
        if epoch % 5 == 0:
            model_save_path = os.path.join(model_save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)

    #Plotting Test loss
    plt.plot(range(len(test_losses)), test_losses, label='Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Losses vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
            
