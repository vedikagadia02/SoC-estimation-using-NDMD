import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from ..models.define_nn_arch import KoopmanNetwork
from ..models.nn_approach_1 import trainAE,testModel
from ..models.nn_approach_1 import trainModel
from ..models.nn_approach_2 import trainModel_2
from sklearn.preprocessing import StandardScaler

# FILES IN FOLLOWING FILE STRUCTURE .\Desktop\SOC-estimation-using-NDMD\paper_implementation

# RUN WITH THIS COMMAND FROM DESKTOP (PARENT DIRECTORY OF SOC_IMPLEMENTATION_NDMD) python -m SoC-estimation-using-NDMD.paper_implementation.main

def save_actual_vs_predicted_plots(data_for_plot, actual_values, predicted_values, epoch, save_folder):
    # actual_values [2000,2]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    actual = actual_values.view(400,499,2)
    predicted = predicted_values.view(400,499,2)
    x1_actual = actual[:,:,0]
    x1_predicted = predicted[:,:,0]
    x2_actual = actual[:,:,1]
    x2_predicted = predicted[:,:,1]
    t_data = data_for_plot[:,:,2]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5)) 
    
    # Plot x1_data vs t_data
    for j in range(x1_actual.shape[0]):
        axs[0].plot(t_data[j,:-1], x1_actual[j,:].cpu().numpy(), label = 'actual', color = 'red')
        axs[0].plot(t_data[j,:-1], x1_predicted[j,:].cpu().numpy(), label = 'predicted', color = 'blue')
        axs[0].set_title('x1_data vs t_data')
        axs[0].set_xlabel('t_data')
        axs[0].set_ylabel('x1_data')

    # Plot x2_data vs t_data
    for j in range(x1_actual.shape[0]):
        axs[1].plot(t_data[j,:-1], x2_actual[j,:].cpu().numpy(), label = 'actual', color = 'red')
        axs[1].plot(t_data[j,:-1], x2_predicted[j,:].cpu().numpy(), label = 'predicted', color = 'blue')
        axs[1].set_title('x2_data vs t_data')
        axs[1].set_xlabel('t_data')
        axs[1].set_ylabel('x2_data')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'plot_x1_epoch_{epoch}.png'))  # Save the plot
    plt.close()  # Close the plot to release memory

def loadData(file_name, ntrain, ntest, train_batch_size, test_batch_size):
    data = np.load(file_name)
    idx = np.arange(0, data.shape[0],1)
    endIdx = data.shape[1]
    np.random.shuffle(idx)
    
    data_for_plot = data[idx[ntrain:ntrain+ntest],:,:]

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
    
    testing_input_data = np.concatenate([testing_data[:, i, :] for i in range(0,endIdx-31,1)], axis=0)
    testing_target_data = np.concatenate([testing_data[:, i:i+30, :] for i in range(1, endIdx-30, 1)], axis=0)

    training_dataset = TensorDataset(torch.Tensor(training_input_data), torch.Tensor(training_target_data))
    training_loader = DataLoader(training_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True)

    testing_dataset = TensorDataset(torch.Tensor(testing_input_data), torch.Tensor(testing_target_data))
    testing_loader = DataLoader(testing_dataset, batch_size = test_batch_size, shuffle = True, pin_memory = True)

    return training_loader, testing_loader, data_for_plot

def print_training_loader_shape(training_loader):
    num_batches = len(training_loader)

    # Shape of a single batch
    for batch in training_loader:
        batch_shape = batch[0].shape
        break

    print(f"Number of batches: {num_batches}")
    print(f"Shape of a single batch: {batch_shape}")

def make_dir(model_dir, koopman_dir):
    model_save_dir = model_dir
    koopman_save_dir = koopman_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)
    if not os.path.exists(koopman_save_dir):
        os.makedirs(koopman_save_dir, exist_ok=True)
    return model_save_dir, koopman_save_dir

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(device))
    
    '''
    CHANGE VALUES HERE
    '''
    dataset = "./SOC-estimation-using-NDMD/paper_implementation/dataset/equation_1/equation_1.npy"
    model_dir = "./SOC-estimation-using-NDMD/paper_implementation/checkpoints"
    koopman_dir = "./SOC-estimation-using-NDMD/paper_implementation/KoopmanPerEpoch"
    plot_dir = "./SOC-estimation-using-NDMD/paper_implementation/plots/test"
    train_batch_size = 128 
    test_batch_size = 16 
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
    epochs_encoder = 4
    '''
    END OF CHANGE VALUES
    '''
    
    exp = 'eqn'
    test_losses = []
    model_save_dir, koopman_save_dir = make_dir(model_dir, koopman_dir)
    training_loader, testing_loader, data_for_plot = loadData(dataset, ntrain, ntest, train_batch_size, test_batch_size)
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
        start_time = time.time()
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

            # Saving the Koopman matrix after every epoch
            koopman_save_path = os.path.join(koopman_save_dir, f"koopman_epoch_{epoch}.txt")
            with open(koopman_save_path, "w") as f:
                for row in koopman:
                    formatted_row = [format(element, '.5f') for element in row]
                    f.write(' '.join(formatted_row) + "\n")
        
        # Saving the model at every 5th epoch
        if epoch % 5 == 0:
            model_save_path = os.path.join(model_save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            
            # Testing and plotting the model at every 5th epoch
            with torch.no_grad():
                test_loss, actual_values, predicted_values = testModel(exp, epoch, device, model, testing_loader, plot_dir)
                test_losses.append(test_loss)
                # save_actual_vs_predicted_plots(data_for_plot, actual_values,predicted_values,epoch, plot_dir)
                print('Epoch {:d}: Testing Loss {:.02f}'.format(epoch, test_loss))
        
        end_time = time.time()  
        elapsed_time = end_time - start_time 
        print('Epoch {:d}: Elapsed time {:.02f} seconds'.format(epoch, elapsed_time))
        
    # Plotting Test loss
    plt.plot(range(len(test_losses)), test_losses, label='Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Losses vs. Epoch')
    plt.show()
            
