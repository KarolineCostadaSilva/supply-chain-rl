import numpy as np
import random
import torch
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from models.environment import SupplyChainEnv
from models.policy import build_ppo2_model

class PPOAgent:
    def __init__(self, env, seed=42):
        self.env = DummyVecEnv([lambda: env])
        self.set_seed(seed)
        self.model = PPO(
            "MlpPolicy", 
            self.env, 
            verbose=1, 
            n_steps=2048, 
            batch_size=64, 
            n_epochs=10, 
            gamma=0.99, 
            learning_rate=0.0001, 
            clip_range=0.2, 
            ent_coef=0.01,
            policy_kwargs= {
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])]},
            seed=seed,
            device="cuda"
        )
        self.log_dir = "results/logs"

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.env is not None:
            self.env.seed(seed)
    
    # def train(self, timesteps, log_dir="results/logs"):
    #     self.model.learn(total_timesteps=timesteps, tb_log_name=log_dir)

    def train(self, timesteps, log_dir="results/logs"):
        episode_rewards = []
        for timestep in range(0, timesteps, self.model.n_steps):
            self.model.learn(total_timesteps=self.model.n_steps, reset_num_timesteps=False)
            rewards = self.evaluate_model()
            episode_rewards.append({'episode': timestep // self.model.n_steps, 'reward': rewards})

        log_df = pd.DataFrame(episode_rewards)
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)

    def evaluate_model(self):
        obs = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward
    
    def predict(self, state):
        action, _states = self.model.predict(state, deterministic=True)
        if np.any(np.isnan(action)):
            raise ValueError("A ação contém valores nan!")
        return action

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

if __name__ == "__main__":
    env = SupplyChainEnv()
    agent = PPOAgent(env)
    agent.train(10000)
    agent.save("results/models/ppo_agent.zip")
