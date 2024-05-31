import yaml
from models.ppo_agent import PPOAgent
from models.environment import SupplyChainEnv

def evaluate(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    env = SupplyChainEnv()
    agent = PPOAgent(env)
    agent.load(config['train_params']['save_path'])

    # Avaliação do agente
    obs = env.reset()
    total_reward = 0
    for _ in range(config['eval_params']['episodes']):
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()

if __name__ == '__main__':
    evaluate('experiments/config.yaml')
