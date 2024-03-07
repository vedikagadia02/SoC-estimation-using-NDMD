import matplotlib.pyplot as plt
import os

def plot_soc(plot_dir, key, epoch, soc_actual, soc_predicted):
    if not os.path.exists(plot_dir + key[4:]):
        os.makedirs(plot_dir + key[4:])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # SoC actual
    axs[0].plot(soc_actual, label='actual', color='red')
    axs[0].set_title('SoC actual')
    axs[0].set_xlabel('t_data')
    axs[0].set_ylabel('soc_data')
    axs[0].legend()

    # SoC predicted
    axs[1].plot(soc_predicted, label='predicted', color='blue')
    axs[1].set_title('SoC predicted')
    axs[1].set_xlabel('t_data')
    axs[1].set_ylabel('soc_data')
    axs[1].legend()

    # SoC actual vs predicted
    axs[2].plot(soc_actual, label='actual', color='red')
    axs[2].plot(soc_predicted, label='predicted', color='blue')
    axs[2].set_title('SoC actual vs predicted')
    axs[2].set_xlabel('t_data')
    axs[2].set_ylabel('soc_data')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir + key[4:], f'epoch_{epoch}_soc.png'))
    plt.close()


def plot_loss(plot_dir, train_losses, test_losses):
    if not os.path.exists(plot_dir + '/loss'):
        os.makedirs(plot_dir + '/loss')
    fig, axs = plt.subplots(len(test_losses) + 1, 1, figsize=(10, 5))
    axs[0].plot(train_losses, label='Train Losses', color='blue')
    axs[0].set_title('Train Losses')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    colors = ['red', 'green', 'purple', 'orange']  # Define colors for different temperatures
    for i, (temp, losses) in enumerate(test_losses.items()):
        axs[i+1].plot(losses, label=f'Test Losses {temp}', color=colors[i])
        axs[i+1].set_title(f'Test Losses {temp}')
        axs[i+1].set_xlabel('Epochs')
        axs[i+1].set_ylabel('Loss')
        axs[i+1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir + '/loss', 'loss_plt.png'))
    plt.close()
