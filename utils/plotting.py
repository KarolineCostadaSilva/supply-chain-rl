import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def plot_scenario(scenario):
    # Carregar os dados
    df = pd.read_csv(f'data/processed/{scenario}.csv')
    
    # Determinar parâmetros
    perturbation_std = int(scenario[1:]) if scenario[1:].isdigit() else 20

    # Configuração dos dados para plotagem
    t = df['Time']
    S = df['Demand'].mean() + df['Demand'].std() * np.sin(2 * np.pi * t / t.max())

    # Criar o gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(t, df['Demand'], 'b.', label='Instance of demands for one retailer', alpha=0.5)
    plt.plot(t, S, 'k-', linewidth=2, label='Sinusoidal function (representing expected values)')
    plt.fill_between(t, S - perturbation_std, S + perturbation_std, color='gray', alpha=0.3, label='Standard deviation of the perturbation')
    
    plt.title(f'Demands for Scenarios {scenario}')
    plt.xlabel('Time step')
    plt.ylabel('Amount of material')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico
    save_path = f'results/plots/{scenario}_demands.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
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
    
    
def plot_scenario_costs(scenarios, results):
    scenario_labels = scenarios
    costs = [result['lp_cost'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(scenario_labels, costs, marker='o', linestyle='-')
    plt.title('Custos Operacionais por Cenário')
    plt.xlabel('Cenário')
    plt.ylabel('Custo Operacional')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/scenario_costs_plot.png')
    plt.show()


if __name__ == '__main__':
    plot_scenario('N20')
    plot_scenario('N60')
