import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os 

def plot_eqn(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir):
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

def plot_soc(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir):
    if not os.path.exists(plot_dir + test_key[4:]):
        os.makedirs(plot_dir + test_key[4:])
    
    soc_actual = actual_values[:,:,-1].to(device)
    soc_predicted = predicted_values[:,:,-1].to(device)

    plt.plot(soc_actual.cpu().numpy(), color='red')
    plt.plot(soc_predicted.cpu().numpy(), color='blue')
    plt.title('SoC actual vs predicted')
    plt.xlabel('t_data')
    plt.ylabel('soc_data')

    # Create custom legend
    red_line = mlines.Line2D([], [], color='red', label='Target')
    blue_line = mlines.Line2D([], [], color='blue', label='Predicted')
    plt.legend(handles=[red_line, blue_line])

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir + test_key[4:], f'plot_epoch_{epoch}_index_{idx}.png'))
    plt.close()

def plot_stock(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir):
    plot_key_path = plot_dir + '/' + test_key[5:]
    if not os.path.exists(plot_key_path):
        os.makedirs(plot_key_path)
    
    stock_actual = actual_values[:,:,-1].to(device)
    stock_predicted = predicted_values[:,:,-1].to(device)

    plt.plot(stock_actual.cpu().numpy(), color='red')
    plt.plot(stock_predicted.cpu().numpy(), color='blue')
    plt.title('TATAMOTORS actual vs predicted')
    plt.xlabel('t_data')
    plt.ylabel('TATAMOTORS_data')

    # Create custom legend
    red_line = mlines.Line2D([], [], color='red', label='Target')
    blue_line = mlines.Line2D([], [], color='blue', label='Predicted')
    plt.legend(handles=[red_line, blue_line])

    plt.tight_layout()
    plt.savefig(os.path.join(plot_key_path, f'epoch_{epoch}_index_{idx}.png'))
    plt.close()
