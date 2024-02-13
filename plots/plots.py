import matplotlib.pyplot as plt
import os 

def plot_eqn(device, epoch, idx, actual_values, predicted_values, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    x1_actual = actual_values[:,:,0].to(device)
    x1_predicted = predicted_values[:,:,0].to(device)
    x2_actual = actual_values[:,:,1].to(device)
    x2_predicted = predicted_values[:,:,1].to(device)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5)) 
    
    # Plot x1_data vs t_data
    for j in range(x1_actual.shape[0]):
        axs[0].plot(x1_actual[j,:].cpu().numpy(), label = 'actual', color = 'red')
        axs[0].plot(x1_predicted[j,:].cpu().numpy(), label = 'predicted', color = 'blue')
        axs[0].set_title('x1 actual vs predicted')
        axs[0].set_xlabel('t_data')
        axs[0].set_ylabel('x1_data')

    # Plot x2_data vs t_data
    for j in range(x1_actual.shape[0]):
        axs[1].plot(x2_actual[j,:].cpu().numpy(), label = 'actual', color = 'red')
        axs[1].plot(x2_predicted[j,:].cpu().numpy(), label = 'predicted', color = 'blue')
        axs[1].set_title('x2 actual vs predicted')
        axs[1].set_xlabel('t_data')
        axs[1].set_ylabel('x2_data')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_epoch_{epoch}_index_{idx}.png'))
    plt.close()

def plot_soh(device, epoch, idx, actual_values, predicted_values, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    soc_actual = actual_values[:,:,0].to(device)
    soc_predicted = predicted_values[:,:,0].to(device)

    plt.plot(soc_actual.cpu().numpy(), label='Target', color='red')
    plt.plot(soc_predicted.cpu().numpy(), label='Predicted', color='blue')
    plt.title('SoC actual vs predicted')
    plt.xlabel('t_data')
    plt.ylabel('soc_data')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_epoch_{epoch}_index_{idx}.png'))
    plt.close()