# Adaptação do código extraido do https://github.com/vfg7/ppo
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_ppo2_model(dimensions, action_space_dim):
    model = keras.Sequential([
        layers.Input(shape=(dimensions,)),  # Usando Input layer
        layers.Reshape((27, 1)),  # Reshape to 2D
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space_dim * 2),  # Output means and log_std for each action
    ])
    return model