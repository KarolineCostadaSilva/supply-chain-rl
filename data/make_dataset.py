import numpy as np
import pandas as pd
import os

def generate_sinusoidal_data(min_val, max_val, z, t, horizon, perturbation_std):
    S = min_val + (max_val - min_val) / 2 * (1 + np.sin(2 * z * t * np.pi / horizon))
    P = np.random.normal(0, perturbation_std, horizon)
    D = np.clip(S + P, min_val, max_val)
    return D

def create_datasets():
    horizon = 360
    min_demand = 100
    max_demand = 300
    z = 1  # Number of peaks in the sinusoidal function
    t = np.arange(horizon)
    
    demand_perturbations = [0, 20, 40, 60]
    for p in demand_perturbations:
        data = generate_sinusoidal_data(min_demand, max_demand, z, t, horizon, p)
        df = pd.DataFrame(data, columns=['demand'])
        os.makedirs('data/processed/', exist_ok=True)
        df.to_csv(f'data/processed/demand_p{p}.csv', index=False)

if __name__ == "__main__":
    create_datasets()
