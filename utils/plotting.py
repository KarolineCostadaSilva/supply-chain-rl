import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_scenario(df, scenario, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time'], df['Demand'], label='Demands')
    plt.title(f'Demands for Scenario {scenario}')
    plt.xlabel('Time step')
    plt.ylabel('Amount of material')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
def plot_results(log_dir, save_path):
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
    scenarios = ['N20', 'N60']
    for scenario in scenarios:
        df = pd.read_csv(f'data/processed/{scenario}.csv')
        plot_scenario(df, scenario, f'results/plots/{scenario}_demands.png')