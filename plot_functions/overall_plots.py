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

def plot_stock(plot_dir, key, epoch,  mm_scale_actual, mm_scale_predicted, iw_scale_actual, iw_scale_predicted):

    plot_key_path = plot_dir + '/' + key[5:]
    if not os.path.exists(plot_key_path):
        os.makedirs(plot_key_path)
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    # stock actual
    axs[0][0].plot(mm_scale_actual, label='actual', color='red')
    axs[0][0].set_title('stock actual')
    axs[0][0].set_xlabel('t_data')
    axs[0][0].set_ylabel('stock_data')
    axs[0][0].legend()

    # stock predicted
    axs[0][1].plot(mm_scale_predicted, label='predicted', color='blue')
    axs[0][1].set_title('stock predicted')
    axs[0][1].set_xlabel('t_data')
    axs[0][1].set_ylabel('stock_data')
    axs[0][1].legend()

    # stock actual vs predicted
    axs[0][2].plot(mm_scale_actual, label='actual', color='red')
    axs[0][2].plot(mm_scale_predicted, label='predicted', color='blue')
    axs[0][2].set_title('stock actual vs predicted')
    axs[0][2].set_xlabel('t_data')
    axs[0][2].set_ylabel('stock_data')
    axs[0][2].legend()

    axs[1][0].plot(iw_scale_actual, label='actual', color='red')
    axs[1][0].set_title('stock actual')
    axs[1][0].set_xlabel('t_data')
    axs[1][0].set_ylabel('stock_data')
    axs[1][0].legend()

    # stock predicted
    axs[1][1].plot(iw_scale_predicted, label='predicted', color='blue')
    axs[1][1].set_title('stock predicted')
    axs[1][1].set_xlabel('t_data')
    axs[1][1].set_ylabel('stock_data')
    axs[1][1].legend()

    # stock actual vs predicted
    axs[1][2].plot(iw_scale_actual, label='actual', color='red')
    axs[1][2].plot(iw_scale_predicted, label='predicted', color='blue')
    axs[1][2].set_title('stock actual vs predicted')
    axs[1][2].set_xlabel('t_data')
    axs[1][2].set_ylabel('stock_data')
    axs[1][2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_key_path, f'test_epoch_{epoch}.png'))
    plt.close()

def plot_loss(plot_dir, key, train_losses, test_losses):
    plot_key_path = plot_dir + '/loss/' + key[5:]
    if not os.path.exists(plot_key_path):
        os.makedirs(plot_key_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    axs[0].plot([t.cpu().item() for t in train_losses], label='Train Losses', color='blue')
    axs[0].set_title('Train Losses')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot([t.cpu().item() for t in test_losses], label='Test Losses', color='red')
    axs[1].set_title('Test Losses')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # colors = ['red', 'green', 'purple', 'orange']  # Define colors for different temperatures
    # for i, (temp, losses) in enumerate(test_losses.items()):
    #     axs[i+1].plot(losses, label=f'Test Losses {temp}', color=colors[i])
    #     axs[i+1].set_title(f'Test Losses {temp}')
    #     axs[i+1].set_xlabel('Epochs')
    #     axs[i+1].set_ylabel('Loss')
    #     axs[i+1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_key_path, 'loss_plt.png'))
    plt.close()

