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
        
        # Função objetivo: minimizar custos de produção, transporte e estoque
        production_costs = self.env.production_costs_suppliers
        transport_costs = self.env.transport_cost
        inventory_costs = self.env.stock_costs
        
        self.model += (pulp.lpSum([production_costs[i] * self.production_vars[i] for i in range(2)]) +
                       pulp.lpSum([transport_costs * self.transport_vars[i] for i in range(12)]) +
                       pulp.lpSum([inventory_costs[i] * self.inventory_vars[i] for i in range(8)]))
        
        # Restrições de capacidade de produção e estoque
        production_capacities = self.env.production_capacities
        for i in range(2):
            self.model += self.production_vars[i] <= production_capacities[i]
        
        stock_capacities = self.env.stock_capacities
        for i in range(8):
            self.model += self.inventory_vars[i] <= stock_capacities[i]
        
        
        # Restrições de balanceamento de material
        # for i in range(8):  # Ajustar para lidar com o número correto de transport_vars e inventory_vars
        #     if i < 2:  # Para os fornecedores
        #         self.model += self.production_vars[i] - self.transport_vars[i * 6] == self.inventory_vars[i]
        #     elif i < 6:  # Para centros de distribuição e varejistas
        #         self.model += pulp.lpSum(self.transport_vars[(i - 2) * 3:(i - 1) * 3]) - pulp.lpSum(self.transport_vars[(i - 1) * 3:i * 3]) == self.inventory_vars[i]
        #     else:  # Para os pontos de venda final
        #         self.model += pulp.lpSum(self.transport_vars[(i - 2) * 3:(i - 1) * 3]) == self.inventory_vars[i]
        for i in range(8):
            self.model += (self.inventory_vars[i] == self.env.demand_data[self.env.current_step])

    def solve(self):
        # Resolução do modelo
        self.model.solve()
        
        # Retorno de ações baseadas na solução do LP
        actions = [var.varValue for var in self.production_vars] + [var.varValue for var in self.transport_vars]
        return actions
