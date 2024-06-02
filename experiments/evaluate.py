import yaml
import torch
import pandas as pd
from models.ppo_agent import PPOAgent
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv
from utils.data_utils import load_data

def evaluate(config_path, scenario):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    demand_data = pd.read_csv(f'data/processed/{scenario}.csv')['Demand'].values
    env = SupplyChainEnv(demand_data=demand_data)
    agent = PPOAgent(env)
    agent.load(config['train_params']['save_path'].replace('.zip', f'_{scenario}.zip'))

    obs = env.reset()
    total_reward = 0
    for _ in range(config['eval_params']['episodes']):
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
    
    print(f"Total reward for scenario {scenario}: {total_reward}")
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU")

    scenarios = ['N0', 'N20', 'N40', 'N60']
    for scenario in scenarios:
        print(f"Evaluating for scenario {scenario}")
        evaluate('experiments/config.yaml', scenario)
