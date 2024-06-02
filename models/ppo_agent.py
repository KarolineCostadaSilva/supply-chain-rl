import numpy as np
import random
import torch
import pandas as pd
import os
import gym
# import tensorflow as tf
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.environment import SupplyChainEnv
from models.neural_networks import ActorNetwork, CriticNetwork

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.network = ActorNetwork(observation_space.shape, features_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.actor = ActorNetwork(observation_space.shape, action_space.shape[0])
        self.critic = CriticNetwork(observation_space.shape)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs):
        actions = self.actor(obs)
        values = self.critic(obs)
        return actions, values

    def _predict(self, obs, deterministic=False):
        actions = self.actor(obs)
        if deterministic:
            actions = torch.tanh(actions)
        return actions

    def evaluate_actions(self, obs, actions):
        value = self.critic(obs)
        log_prob = self.actor(obs).log_prob(actions)
        return value, log_prob, torch.zeros_like(log_prob)

class PPOAgent:
    def __init__(self, env, seed=42):
        self.env = DummyVecEnv([lambda: env])
        # self.device = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
        
        policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=64),
            # "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        }
        self.model = PPO(
            CustomActorCriticPolicy, 
            self.env, 
            verbose=1, 
            n_steps=2048, 
            batch_size=64, 
            n_epochs=10, 
            gamma=0.99, 
            learning_rate=0.0001, 
            clip_range=0.2, 
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            seed=seed
        )
        self.log_dir = "results/logs"
        self.set_seed(seed)

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
        self.model = PPO.load(path, env=self.env)

if __name__ == "__main__":
    scenarios = ['N0', 'N20', 'N40', 'N60']
    for scenario in scenarios:
        demand_data = pd.read_csv(f'data/processed/{scenario}.csv')['demand'].values
        env = SupplyChainEnv(demand_data=demand_data)
        agent = PPOAgent(env)
        agent.train(10000)
        agent.save(f"results/models/ppo_agent_{scenario}.zip")
