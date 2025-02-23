import torch
import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import CnnPolicy

class CustomCNNMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomCNNMLP, self).__init__(observation_space, features_dim)
        
        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calcular a dimensão de saída da CNN
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(observations)
        mlp_features = self.mlp(cnn_features)
        return mlp_features

class CNN_MLP_Policy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CNN_MLP_Policy, self).__init__(*args, **kwargs, features_extractor_class=CustomCNNMLP)