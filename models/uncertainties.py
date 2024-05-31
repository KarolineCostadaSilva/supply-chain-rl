# Adaptação do código extraido do https://github.com/vfg7/ppo
import random
import math
import numpy as np

def stochastic_demand(min_val, max_val, freq, t, total_timesteps, mean, std):
    s = sinusoidal(min_val, max_val, freq, total_timesteps, t)
    d = disturbance(mean, std)
    return clip(s + d, max_val, min_val)

def clip(x, max_val, min_val):
    return min_val if x < min_val else max_val if x > max_val else x

def sinusoidal(min_val, max_val, freq, total_timesteps, given_timestep=None):
    given_timestep = given_timestep if given_timestep != 0 else 1
    return min_val + (max_val - min_val) / 2 * (1 + math.sin(2 * freq * given_timestep * math.pi / total_timesteps))

def disturbance(mean, std_dev):
    return np.random.normal(mean, std_dev)

def normalize(value, min_val, max_val):
    if min_val == max_val:
        raise ValueError("min_val and max_val must be diferentes for normalization.")
    return (value - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val):
    return min_val + value * (max_val - min_val)
