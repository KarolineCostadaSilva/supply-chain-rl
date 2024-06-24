import numpy as np
import pandas as pd
import os

def generate_sinusoidal_data(min_val, max_val, z, t, horizon, perturbation_std, perturbation_range=None, uniform_pert=False):
    S = min_val + (max_val - min_val) / 2 * (1 + np.sin(2 * z * t * np.pi / horizon))
    if uniform_pert:
        P = np.random.uniform(perturbation_range[0], perturbation_range[1], horizon)
    else:
        P = np.random.normal(0, perturbation_std, horizon)
    D = S + P
    return S, D, P

def create_datasets():
    horizon = 360
    z = 1  # Number of peaks in the sinusoidal function
    t = np.arange(horizon)
    
    scenarios = {
        'N0': (100, 300, 0, None, False),
        'N20': (100, 300, 20, None, False),
        'N40': (100, 300, 40, None, False),
        'N60': (100, 300, 60, None, False),

        'N0cl': (100, 300, 0, None, False),
        'N20cl': (100, 300, 20, None, False),
        'N40cl': (100, 300, 40, None, False),
        'N60cl': (100, 300, 60, None, False),

        'rN0': (100, 300, 0, [-200, 200], True),
        'rN50': (100, 300, 50, [-200, 200], True),
        'rN100': (100, 300, 100, [-200, 200], True),
        'rN200': (100, 300, 200, [-200, 200], True),

        'rN0cl': (100, 330, 0, [-220, 220], True),
        'rN50cl': (100, 330, 50, [-220, 220], True),
        'rN100cl': (100, 330, 100, [-220, 220], True),
        'rN200cl': (100, 330, 220, [-220, 220], True),

        'N20stc': (100, 300, 20, None, False)  # This requires special handling in simulation for stock costs
    }
    directory = 'data/processed/'
    os.makedirs(directory, exist_ok=True)
    
    for scenario, (min_demand, max_demand, perturbation_std, perturbation_range, uniform_pert) in scenarios.items():
        _, demand, _ = generate_sinusoidal_data(min_demand, max_demand, z, t, horizon, perturbation_std, perturbation_range, uniform_pert)
        df = pd.DataFrame({'Time': t, 'Demand': demand})
        df.to_csv(f'{directory}{scenario}.csv', index=False)

if __name__ == "__main__":
    create_datasets()
