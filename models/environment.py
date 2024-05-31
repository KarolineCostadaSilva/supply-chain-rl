import gym
from gym import spaces
import numpy as np
from models.uncertainties import stochastic_demand

class SupplyChainEnv(gym.Env):
    def __init__(self, max_steps=1000, limiar_inventario=10):
        super(SupplyChainEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(27,), dtype=np.float32)
        self.max_steps = max_steps
        self.limiar_inventario = limiar_inventario
        self.current_step = 0
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
        self.demand = np.array([stochastic_demand(50, 150, 0.1, t, 360, 100, 20) for t in range(3)]) # 3 variáveis de demanda
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
        self.demand = np.array([stochastic_demand(50, 150, 0.1, t, 360, 100, 20) for t in range(3)])
        return self.get_observation()

    def calculate_reward(self, state, action):
        # Calcular custos de inventário, produção e transporte
        inventory_cost = np.sum(self.inventory)
        production_cost = np.sum(self.production)
        transport_cost = np.sum(self.transport)
        unmet_demand_penalty = np.sum(np.maximum(self.demand - self.inventory[:3], 0))
        reward = -(inventory_cost + production_cost + transport_cost + unmet_demand_penalty)
        return reward

    def check_done(self, state):
        # Verificar se a simulação deve terminar
        # Condição 1: Número máximo de etapas atingido
        if self.current_step >= self.max_steps:
            return True
        
        # Condição 2: Inventário abaixo de um limiar por um período prolongado
        limiar_inventario = self.limiar_inventario
        if np.any(self.inventory < limiar_inventario):
            return True
        
        return False
    
    def get_observation(self):
        # Retornar a observação do estado atual
        return np.concatenate([self.inventory, self.production, self.transport, self.demand])
