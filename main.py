import torch
from experiments.train_ppo import train_ppo as train_ppo
from experiments.evaluate import evaluate
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU")
    
    scenarios = ['N0', 'N20', 'N40', 'N60']
    for scenario in scenarios:
        print(f"Training for scenario {scenario}")
        train_ppo('experiments/config.yaml', scenario)
        print(f"Evaluating for scenario {scenario}")
        evaluate('experiments/config.yaml', scenario)
