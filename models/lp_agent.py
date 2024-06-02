import numpy as np
import pulp

class LPAgent:
    def __init__(self, env):
        self.env = env
        self.setup_model()

    def setup_model(self):
        self.model = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
        
        # Definir as variáveis de decisão para produção, transporte e estoque
        self.production_vars = [pulp.LpVariable(f'production_{i}', lowBound=0) for i in range(2)]
        self.transport_vars = [pulp.LpVariable(f'transport_{i}', lowBound=0) for i in range(12)]
        self.inventory_vars = [pulp.LpVariable(f'inventory_{i}', lowBound=0) for i in range(8)]
        
        # Definir função objetivo (minimizar custos)
        production_costs = self.env.production_costs_suppliers
        transport_costs = self.env.transport_cost
        inventory_costs = self.env.stock_costs
        
        self.model += (pulp.lpSum([production_costs[i] * self.production_vars[i] for i in range(2)]) +
                       pulp.lpSum([transport_costs * self.transport_vars[i] for i in range(12)]) +
                       pulp.lpSum([inventory_costs[i] * self.inventory_vars[i] for i in range(8)]))
        
        # Definir restrições de capacidade de produção
        production_capacities = self.env.production_capacities
        for i in range(2):
            self.model += self.production_vars[i] <= production_capacities[i]
        
        # Definir restrições de capacidade de estoque
        stock_capacities = self.env.stock_capacities
        for i in range(8):
            self.model += self.inventory_vars[i] <= stock_capacities[i]
        
        # Definir restrições de balanceamento de material
        # self.model += (self.production_vars[0] - self.transport_vars[0] == self.inventory_vars[0])  # Nó 1
        # self.model += (self.transport_vars[0] - self.transport_vars[1] == self.inventory_vars[1])   # Nó 2
        # self.model += (self.production_vars[1] - self.transport_vars[2] == self.inventory_vars[2])  # Nó 3
        # self.model += (self.transport_vars[2] - self.transport_vars[3] == self.inventory_vars[3])   # Nó 4
        # self.model += (self.transport_vars[1] - self.transport_vars[4] == self.inventory_vars[4])   # Nó 5
        # self.model += (self.transport_vars[4] - self.transport_vars[5] == self.inventory_vars[5])   # Nó 6
        # self.model += (self.transport_vars[3] - self.transport_vars[6] == self.inventory_vars[6])   # Nó 7
        # self.model += (self.transport_vars[5] - self.transport_vars[7] == self.inventory_vars[7])   # Nó 8
        # Definir restrições de balanceamento de material conforme a metodologia do artigo
        for i in range(1, 9):
            self.model += (self.production_vars[i % 2] - self.transport_vars[i - 1] == self.inventory_vars[i - 1])

    def solve(self):
        # Resolução do modelo
        self.model.solve()
        
        # Retorno de ações baseadas na solução do LP
        actions = [var.varValue for var in self.production_vars] + [var.varValue for var in self.transport_vars]
        return actions
