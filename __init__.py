import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Parâmetros Gerais
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1000
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializando o ambiente
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
print(env)
input_shape = (4, 84, 84)  # Usando 4 frames empilhados como entrada
num_actions = env.action_space.n

# Rede Neural DQN
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.to(device)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_input_dim = self.feature_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def feature_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.float() / 255.0  # Normalizando
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Memória de Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # Move tensors to CPU before converting to numpy arrays
        state = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in state])
        next_state = np.array([ns.cpu().numpy() if isinstance(ns, torch.Tensor) else ns for ns in next_state])
        return state, action, reward, next_state, done

    
    def __len__(self):
        return len(self.buffer)

# Agente DQN
class Agent:
    def __init__(self, input_shape, num_actions):
        self.model = DQN(input_shape, num_actions).to('cuda')
        self.target_model = DQN(input_shape, num_actions).to('cuda')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.num_actions = num_actions
        self.steps = 0
        

    def select_action(self, state):
        self.epsilon = max(EPSILON_END, EPSILON_START - self.steps / EPSILON_DECAY)
        self.steps += 1
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float32).to('cuda')
            with torch.no_grad():
                return self.model(state).argmax(dim=1).item()
    
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        states = torch.tensor(states, dtype=torch.float32).to('cuda')
        next_states = torch.tensor(next_states, dtype=torch.float32).to('cuda')
        actions = torch.tensor(actions).to('cuda')
        rewards = torch.tensor(rewards).to('cuda')
        dones = torch.tensor(dones, dtype=torch.float32).to('cuda')
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Inicializando o Agente
agent = Agent(input_shape, num_actions)

# Loop de Treinamento
num_episodes = 1000
input_shape = (4, 84, 84)  # Exemplo, ajuste conforme o seu ambiente
num_actions = env.action_space.n
model = DQN(input_shape, num_actions)
#model.load_state_dict(torch.load("dqn_model.pth"))
model.to(device)
model.eval()

for episode in range(num_episodes):
    observation, info = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    done = False
    total_reward = 0
    
    while not done:
        if episode / 50 == 1:  # Salva a cada 50 episódios
          torch.save(model.state_dict(), f"dqn_model_episode_{episode}.pth")

        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Armazenar transição na memória de replay
        agent.replay_buffer.push(observation, action, reward, next_observation, done)
        
        # Treinamento do DQN
        agent.train()
        
        observation = next_observation
        total_reward += reward
        
        if done:
            print(f"Episode {episode+1} - Total Reward: {total_reward}")
            break

    # Atualização periódica da Target Network
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

env.close()
