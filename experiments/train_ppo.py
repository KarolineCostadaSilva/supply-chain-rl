import yaml
import tensorflow as tf
from models.ppo_agent import PPOAgent
from models.environment import SupplyChainEnv
from utils.plotting import plot_results

def train_ppo(config_path):
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponível. Usando GPU.")
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        device = '/GPU:0'
    else:
        print("GPU não disponível. Usando CPU.")
        device = '/CPU:0'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    with tf.device(device):
        env = SupplyChainEnv()
        agent = PPOAgent(env, seed=42)

        agent.train(config['train_params']['timesteps'], log_dir="results/logs")
        agent.save(config['train_params']['save_path'])

    plot_results("results/logs", "results/plots/training_plot.png")
    
if __name__ == '__main__':
    train_ppo('experiments/config.yaml')
