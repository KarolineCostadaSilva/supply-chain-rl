import tensorflow as tf
from experiments.train_ppo import train_ppo as train_ppo
# from experiments.evaluate import evaluate

if __name__ == "__main__":
    train_ppo('experiments/config.yaml')
    # evaluate('experiments/config.yaml')
