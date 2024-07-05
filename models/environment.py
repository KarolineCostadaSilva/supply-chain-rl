import gym
from gym import spaces
import numpy as np
import pandas as pd

class SupplyChainEnv(gym.Env):
    def __init__(self, scenario_name, max_steps=360, limiar_inventario=10):
        super(SupplyChainEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(23,), dtype=np.float32)
        self.max_steps = max_steps
        self.limiar_inventario = limiar_inventario
        self.current_step = 0
        # self.q = 8
        # self.fn = 1
        # self.rn = 3
        self.production_costs_suppliers = [6, 4]
        self.production_costs_factories = [12, 10]
        self.transport_cost = 2
        self.penalty_costs = 10
        self.unmet_demand_cost = 216
        self.production_capacities = [600, 840]
        self.processing_capacities = [840, 960]
        self.stock_capacities = [6400, 7200, 1600, 1800] * 2
        self.initial_stock_levels = 800
        self.material_available = [600, 840] * 4
        self.material_arrival_times = [240, 240] * 4
        # self.material_arrival_times_factories = [240, 240] * 4
        self.stock_costs = [1] * 8  # Custos de inventário para todos os nós
        self.demand_data = self.load_demand_data(scenario_name)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        # Reinicia o estado do ambiente no início de cada novo episódio
        self.current_step = 0
        self.inventory = np.full((8,), self.initial_stock_levels)  # Assumindo 8 nós
        self.production = np.zeros((2,))  # Assumindo 2 fornecedores
        self.transport = np.zeros((12,))  # Assumindo 14 links de transporte
        self.demand = self.demand_data[self.current_step]  # Amostra a demanda inicial
        return self.get_observation()

    def load_demand_data(self, scenario_name):
        # Carrega e retorna os dados de demanda processados considerando incertezas e sazonalidade
        demand_df = pd.read_csv(f'data/processed/{scenario_name}.csv')
        return demand_df['Demand'].values
    
    def step(self, action):
        # Aplica ações de produção e transporte
        self.apply_actions(action)
        # Atualiza o inventário com base na produção, transporte e demanda atual
        self.update_inventory()
        # Calcula a recompensa com base nos custos de operação, estoque e penalidades
        state = np.concatenate([self.inventory, self.production, self.transport, [self.demand]])
        reward = self.calculate_reward(state, action)
        # Verifica se o episódio terminou (por exemplo, número máximo de passos alcançado)
        done = self.check_done()
        # Amostra nova demanda para o próximo período
        self.demand = self.sample_demand()
        # Atualiza o estado para o próximo passo
        self.current_step += 1
        # Coleta informações do estado atual para retorno
        self.state = self.get_observation()
        return self.state, reward, done, {}

    def initialize_state(self):
        # Inicializar o estado com valores adequados
        self.inventory = np.full((8,), self.initial_stock_levels)  # 8 nós na cadeia de suprimentos
        self.production = np.zeros((2,))  # 2 fornecedores
        self.transport = np.zeros((12,))  # 14 fluxos de transporte
        self.demand = self.demand_data[self.current_step]  # Usando os dados de demanda carregados
        return self.get_observation()

    def apply_actions(self, action):
        # Verifique se o tamanho da ação corresponde ao esperado
        if len(action) != 14:
            raise ValueError(f"Expected 14 actions, received {len(action)}")
        
        # Aplica ações de produção nos dois primeiros índices de ação
        self.production = np.clip(self.production + action[:2], 0, self.production_capacities)
        
        # Aplica ações de transporte nos 12 índices seguintes
        self.transport = np.clip(self.transport + action[2:], 0, np.inf)  # Sem limite superior claro para o transporte

    def update_inventory(self):
        # Atualiza o inventário com base na produção e transporte
        # Ajusta os estoques com base na produção e envios
        self.inventory[:2] = np.clip(self.inventory[:2] + self.production - self.transport[:2], 0, self.stock_capacities[:2])
        for i in range(2, 8):
            self.inventory[i] = np.clip(self.inventory[i] + self.transport[i-2], 0, self.stock_capacities[i])

    def calculate_reward(self, state, action):
        # Extraia o inventário, a produção e o transporte do estado
        inventory, production, transport = state[:8], state[8:10], state[10:22]
        # Calcule os custos de produção influenciados pelas ações
        production_action = action[:2]  # Assume que as primeiras duas ações são para produção
        production_cost = np.sum(production_action * np.array(self.production_costs_suppliers))
        # Calcule os custos de transporte influenciados pelas ações
        transport_action = action[2:]  # Assume que as ações restantes são para transporte
        transport_cost = np.sum(transport_action * self.transport_cost)
        # Calcule os custos de inventário
        inventory_cost = np.sum(inventory * np.array(self.stock_costs))
        # Calcule as penalidades por demanda não atendida
        unmet_demand_penalty = np.sum(np.maximum(self.demand - inventory[:3], 0) * self.unmet_demand_cost)
        # A recompensa é a negativa da soma de todos os custos e penalidades
        reward = -(inventory_cost + production_cost + transport_cost + unmet_demand_penalty)
        return reward

    def check_done(self, state):
        # Verificar se a simulação deve terminar
        # Condição 1: Número máximo de etapas atingido
        if self.current_step >= self.max_steps:
            return True
        # Condição 2: Inventário abaixo de um limiar por um período prolongado
        if np.any(self.inventory < self.limiar_inventario):
            return True
        return False
    
    def get_observation(self):
        # Retornar a observação do estado atual
        return np.concatenate([self.inventory, self.production, self.transport, [self.demand]])
    
    def sample_demand(self):
        # Retorna uma demanda amostrada para o próximo período
        if self.current_step < len(self.demand_data) - 1:
            return self.demand_data[self.current_step + 1]
        else:
            # Opção para recomeçar a demanda do início ou continuar com a última demanda conhecida
            return self.demand_data[-1]

