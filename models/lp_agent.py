import numpy as np
import pulp

class LPAgent:
    def __init__(self, env):
        self.env = env
        self.setup_model()

    def setup_model(self):
        self.model = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
        
        # Defining decision variables for production, transportation, and inventory
        self.production_vars = [pulp.LpVariable(f'production_{i}', lowBound=0) for i in range(2)]
        self.transport_vars = [pulp.LpVariable(f'transport_{i}', lowBound=0) for i in range(12)]
        self.inventory_vars = [pulp.LpVariable(f'inventory_{i}', lowBound=0) for i in range(8)]
        
        # Objective function: Minimizing production, transportation, and inventory costs
        production_costs = self.env.production_costs_suppliers
        transport_costs = self.env.transport_cost
        inventory_costs = self.env.stock_costs
        
        self.model += (pulp.lpSum([production_costs[i] * self.production_vars[i] for i in range(2)]) +
                       pulp.lpSum([transport_costs * self.transport_vars[i] for i in range(12)]) +
                       pulp.lpSum([inventory_costs[i] * self.inventory_vars[i] for i in range(8)]))
        
        # Capacity constraints for production and inventory
        production_capacities = self.env.production_capacities
        for i in range(2):
            self.model += self.production_vars[i] <= production_capacities[i]
        
        stock_capacities = self.env.stock_capacities
        for i in range(8):
            self.model += self.inventory_vars[i] <= stock_capacities[i]
        
        # Material balance constraints adjusted for transportation and production logic
        # Suppliers
        for i in range(2):  # Para fornecedores
            self.model += self.production_vars[i] - pulp.lpSum(self.transport_vars[i*6:(i+1)*6]) == self.inventory_vars[i]

        for i in range(2, 8):  # Para centros de distribuição e varejistas
            received = pulp.lpSum(self.transport_vars[(i-2)*3:(i-1)*3]) if i > 2 else 0
            sent = pulp.lpSum(self.transport_vars[(i-1)*3:i*3])
            # Acesso seguro ao elemento de demanda considerando que 'demand' é um array
            # current_demand = self.env.demand_data[self.env.current_step] if i-2 >= 0 else 0
            current_demand = self.env.demand_data[self.env.current_step % len(self.env.demand_data)] if i-2 >= 0 else 0
            self.model += (received + self.inventory_vars[i-1] - sent - current_demand) == self.inventory_vars[i]

    def solve(self):
        # Impressões de depuração antes de resolver o LP
        print("Custos de Produção:", self.env.production_costs_suppliers)
        print("Custos de Transporte:", self.env.transport_cost)
        print("Custos de Inventário:", self.env.stock_costs)
        print("Capacidades de Produção:", self.env.production_capacities)
        print("Capacidades de Estoque:", self.env.stock_capacities)
        for i, var in enumerate(self.production_vars + self.transport_vars + self.inventory_vars):
            print(f"Variável {i}: {var.name}, Limite Inferior: {var.lowBound}, Limite Superior: {var.upBound}")

        # Resolver o modelo
        self.model.solve()
        
        # Impressões de depuração após resolver o LP
        print("Solução do modelo:")
        for v in self.production_vars + self.transport_vars + self.inventory_vars:
            print(f"{v.name} = {v.varValue}")

        # Verificar o status do solver
        print("Status do Solver:", pulp.LpStatus[self.model.status])

        # Retorno de ações baseadas na solução do LP
        actions = [var.varValue for var in self.production_vars] + [var.varValue for var in self.transport_vars]
        return actions
