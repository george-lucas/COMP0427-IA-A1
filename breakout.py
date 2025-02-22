import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import numpy as np
from collections import defaultdict
import random

# Implementação básica do MCTS
class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = []  # Inicializa como uma lista vazia

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_child(self, exploration_weight=1.0):
        # Seleciona o filho com o maior UCB (Upper Confidence Bound)
        return max(self.children, key=lambda c: c.value / (c.visits + 1e-6) + 
               exploration_weight * np.sqrt(2 * np.log(self.visits + 1) / (c.visits + 1e-6)))

    def expand(self, action):
        # Expande um nó com uma ação não tentada
        child = MCTSNode(parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, reward):
        # Atualiza o valor do nó com a recompensa obtida
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, model, env, num_simulations=50, exploration_weight=1.0):
        self.model = model
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def search(self, observation):
        root = MCTSNode()

        for _ in range(self.num_simulations):
            node = root
            sim_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
            sim_env = AtariWrapper(sim_env)  # Aplica os wrappers de pré-processamento
            sim_env.reset()

            # Clona o estado do ambiente original
            if hasattr(self.env, 'envs'):  # Verifica se é um DummyVecEnv
                original_env = self.env.envs[0]  # Acessa o ambiente subjacente
                sim_env.unwrapped.ale.restoreState(original_env.unwrapped.ale.cloneState())
            else:
                sim_env.unwrapped.ale.restoreState(self.env.unwrapped.ale.cloneState())

            # Seleção
            while node.is_fully_expanded() and node.children:
                node = node.select_child(self.exploration_weight)
                observation, _, _, _, _ = sim_env.step(node.action)

            # Expansão
            if not node.untried_actions:  # Se não houver ações não tentadas
                node.untried_actions = list(range(sim_env.action_space.n))  # Inicializa com todas as ações possíveis
            action = random.choice(node.untried_actions)
            child = node.expand(action)
            observation, _, _, _, _ = sim_env.step(action)

            # Simulação
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = sim_env.step(action)
                total_reward += reward
                done = terminated or truncated

            # Retropropagação
            while child is not None:
                child.update(total_reward)
                child = child.parent

        # Escolhe a ação mais visitada
        if root.children:
            return max(root.children, key=lambda c: c.visits).action
        else:
            return random.choice(range(self.env.action_space.n))  # Fallback: ação aleatória

# Configurações
env_name = 'ALE/Breakout-v5'
total_timesteps = 100000  # Número de passos de treino
save_path = './models/dqn_breakout'

# Crie o ambiente do Atari 2600 com pré-processamento
env = make_atari_env(env_name, n_envs=1, seed=42)

# Defina um callback para salvar checkpoints do modelo
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,
                                         name_prefix='dqn_breakout_checkpoint')

# Inicialize o agente DQN
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=0.0001,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    tau=0.1,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="./tensorboard/dqn_breakout/"
)

# Inicialize o MCTS
mcts = MCTS(model, env, num_simulations=50)

# Treine o agente com MCTS
for _ in range(total_timesteps):
    observation = env.reset()
    done = False
    
    while not done:
        # Usa o MCTS para escolher a ação
        action = mcts.search(observation[0])  # Acessa a observação do primeiro ambiente
        next_observation, reward, done, info = env.step([action])  # Passa a ação como uma lista

        # Ajusta o formato das observações para (1, 84, 84)
        observation_reshaped = np.transpose(observation[0], (2, 0, 1))  # Transforma (84, 84, 1) em (1, 84, 84)
        next_observation_reshaped = np.transpose(next_observation[0], (2, 0, 1))  # Transforma (84, 84, 1) em (1, 84, 84)

        # Armazena a experiência no buffer de replay do DQN
        model.replay_buffer.add(
            observation_reshaped,  # Observação atual (formato correto)
            next_observation_reshaped,  # Próxima observação (formato correto)
            np.array([action]),  # Ação (convertida para array)
            np.array([reward]),  # Recompensa (convertida para array)
            np.array([done]),  # Done (convertido para array)
            info  # Infos (dicionário)
        )

        observation = next_observation

        # Treina o modelo DQN
        model.train()

# Salve o modelo treinado
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")