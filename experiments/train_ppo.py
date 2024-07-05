import yaml
import pandas as pd
from models.ppo_agent import PPOAgent
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv
from utils.plotting import plot_results
from utils.data_utils import load_data

def train_ppo(config_path, scenario):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    env = SupplyChainEnv(scenario_name=scenario)
    agent = PPOAgent(env, seed=42)
        
    agent.train(config['train_params']['timesteps'], log_dir=f"results/logs/{scenario}")
    agent.save(f"results/models/ppo_agent_{scenario}.zip")
    
    plot_results(f"results/logs/{scenario}", f"results/plots/training_plot_{scenario}.png")
 
# if __name__ == '__main__':
#     scenarios = ['N0', 'N20', 'N40', 'N60']
#     for scenario in scenarios:
#         print(f"Training for scenario {scenario}")
#         train_ppo('experiments/config.yaml', scenario)
