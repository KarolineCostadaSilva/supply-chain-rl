import yaml
import torch
import pandas as pd
import numpy as np
from models.ppo_agent import PPOAgent
from models.lp_agent import LPAgent
from models.environment import SupplyChainEnv
from utils.data_utils import load_data

def evaluate(scenario):
    # Carregar dados de demanda
    demand_data = pd.read_csv(f'data/processed/{scenario}.csv')['Demand'].values
    env = SupplyChainEnv(scenario)
    
    # Criação e avaliação do LP Agent
    lp_agent = LPAgent(env)
    lp_actions = lp_agent.solve()

    # Capturar os custos do LP Agent
    lp_cost = sum(lp_actions)  

    results = {
        'scenario': scenario,
        'lp_cost': lp_cost
    }
    return results
