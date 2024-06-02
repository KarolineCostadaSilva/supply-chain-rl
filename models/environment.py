import gym
from gym import spaces
import numpy as np

class SupplyChainEnv(gym.Env):
    def __init__(self, demand_data, max_steps=360, limiar_inventario=10):
        super(SupplyChainEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.float32)
        self.max_steps = max_steps
        self.limiar_inventario = limiar_inventario
        self.current_step = 0
        self.q = 8
        self.fn = 1
        self.rn = 3
        self.production_costs_suppliers = [6, 4]
        self.production_costs_factories = [12, 10]
        self.transport_cost = 2
        self.penalty_costs = 10
        self.unmet_demand_cost = 216
        self.production_capacities = [600, 840, 600, 840]
        self.processing_capacities = [840, 960]
        self.stock_capacities = [6400, 7200, 1600, 1800]
        self.initial_stock_levels = 800
        self.material_available = [600, 840]
        self.material_arrival_times = [600, 840]
        self.material_arrival_times_factories = [240, 240]
        self.stock_costs = [1] * 8  # Custos de inventário para todos os nós
        self.demand_data = demand_data
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.state = self.initialize_state()
        self.current_step = 0
        return self.state

    def step(self, action):
        # Implementação da lógica de transição de estado baseada nas ações
        self.state = self.transition_state(action)
        reward = self.calculate_reward(self.state, action)
        done = self.check_done(self.state)
        self.current_step += 1
        return self.state, reward, done, {}

    def initialize_state(self):
        # Inicializar o estado com valores adequados
        self.inventory = np.zeros((8,))  # 8 nós na cadeia de suprimentos
        self.production = np.zeros((2,))  # 2 fornecedores
        self.transport = np.zeros((14,))  # 14 fluxos de transporte
        # self.demand = np.array([self.demand_data[self.current_step]])  # Usando os dados de demanda carregados
        self.demand = self.demand_data[self.current_step]
        return self.get_observation()

    def transition_state(self, action):
        # Verificar tamanhos das ações
        assert len(action) == 14, f"Tamanho da ação esperado: 14, mas recebido: {len(action)}"

        # Transição de estado baseada na ação
        # Ações de produção (2)
        self.production += action[:2]
        # Ações de transporte (12)
        self.transport[:12] += action[2:14]
        # Atualizar inventário com base na produção e transporte
        self.inventory[:2] += self.production
        self.inventory[2:8] -= self.transport[:6]
        self.inventory[2:8] += self.transport[6:12]
        # Atualizar demandas (usando demanda estocástica)
        if self.current_step < len(self.demand_data):
            # self.demand = np.array([self.demand_data[self.current_step]])
            self.demand = self.demand_data[self.current_step]
        return self.get_observation()

    def calculate_reward(self, state, action):
        # Calcular custos de inventário, produção e transporte
        inventory_cost = np.sum(self.inventory * np.array(self.stock_costs))
        production_cost = np.sum(self.production * np.array(self.production_costs_suppliers))
        transport_cost = np.sum(self.transport * self.transport_cost)
        unmet_demand_penalty = np.sum(np.maximum(self.demand - self.inventory[:3], 0) * self.unmet_demand_cost)
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
