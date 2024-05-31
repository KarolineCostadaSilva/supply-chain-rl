import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_results(log_dir, save_path):
    # Lendo o arquivo CSV no diretório de logs
    log_file_path = os.path.join(log_dir, 'training_log.csv')
    log_data = pd.read_csv(log_file_path)
    
    episodes = log_data['episode']
    rewards = log_data['reward']

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Rewards')
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
if __name__ == '__main__':
    plot_results("results/logs", "results/plots/training_plot.png")