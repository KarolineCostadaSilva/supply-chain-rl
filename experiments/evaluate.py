import yaml
import torch
import pandas as pd
import numpy as np
from models.ppo_agent import PPOAgent
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv
from utils.data_utils import load_data

def evaluate(scenario):
    # Carregar configurações
    with open('experiments/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Carregar dados de demanda
    demand_data = pd.read_csv(f'data/processed/{scenario}.csv')['Demand'].values
    env = SupplyChainEnv(demand_data=demand_data)

    # Criação e avaliação do LP Agent
    lp_agent = LPAgent(env)
    lp_actions = lp_agent.solve()
    lp_cost = sum(lp_actions)  # Capturar os custos do LP Agent

    # Criação e avaliação do PPO Agent
    ppo_agent = PPOAgent(env, seed=config['seed'])
    ppo_agent.load(f"results/models/ppo_agent_{scenario}.zip")
    ppo_rewards = ppo_agent.evaluate_model()  # Avaliação do modelo PPO

    results = {
        'scenario': scenario,
        'lp_cost': lp_cost,
        'ppo_rewards': ppo_rewards
    }
    return results
