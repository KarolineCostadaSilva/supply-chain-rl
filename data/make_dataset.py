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
    z = 1  # Number of peaks in the sinusoidal function
    t = np.arange(horizon)
    
    scenarios = {
        'N0': (100, 300, 0),
        'N20': (100, 300, 20),
        'N40': (100, 300, 40),
        'N60': (100, 300, 60)
    }
    for scenario, (min_demand, max_demand, perturbation) in scenarios.items():
        data = generate_sinusoidal_data(min_demand, max_demand, z, t, horizon, perturbation)
        df = pd.DataFrame({'Time': t, 'Demand': data})
        os.makedirs('data/processed/', exist_ok=True)
        df.to_csv(f'data/processed/{scenario}.csv', index=False)

if __name__ == "__main__":
    create_datasets()
