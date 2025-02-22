import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
import random

# Função MCTS (Monte Carlo Tree Search)
class MCTS:
    def __init__(self, env, model, simulations=10):
        self.env = env
        self.model = model
        self.simulations = simulations

    def simulate(self, state):
        # Simula uma partida a partir de um estado
        best_action = None
        best_reward = -float('inf')
        
        for _ in range(self.simulations):
            # Reset do ambiente para um novo episódio de simulação
            current_state = state
            total_reward = 0
            done = False
            
            while not done:
                action = self.model.predict(current_state, deterministic=True)[0]  # Ação escolhida pela DQN
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                current_state = new_state
            
            # Se encontramos uma ação que dá melhor recompensa, selecionamos ela
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        return best_action


# Configurações
env_name = 'ALE/Breakout-v5'
total_timesteps = 10000  # Número de passos de treino
save_path = './models/dqn_breakout'

# Crie o ambiente do Atari 2600
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

# Inicializar MCTS
mcts = MCTS(env, model, simulations=10)  # Definindo um número de simulações

# Associar o modelo ao callback
checkpoint_callback.model = model

# Loop de treinamento personalizado
obs = env.reset()  # Atualizado para `env.reset()` retornar o estado corretamente
for step in range(total_timesteps):
    # Escolha da ação: alternar entre DQN e MCTS
    if np.random.rand() < 0.1:  # 10% de chance de usar MCTS
        action = mcts.simulate(obs)  # Ação escolhida pelo MCTS
    else:
        action, _states = model.predict(obs, deterministic=False)  # Ação escolhida pelo DQN

    # Execute a ação no ambiente
    new_obs, reward, done, info = env.step(action)

    # Verificar se a observação precisa ser ajustada
    if new_obs.shape != (1, 84, 84, 1):  # Verifica se a observação tem o formato esperado
        new_obs = np.repeat(new_obs, 4, axis=-1)  # Repetir as 4 últimas dimensões para ajustar o formato

    # Reshape observations to match the replay buffer's expected shape
    obs_reshaped = np.transpose(obs, (0, 3, 1, 2))  # Convert (1, 84, 84, 1) to (1, 1, 84, 84)
    new_obs_reshaped = np.transpose(new_obs, (0, 3, 1, 2))  # Convert (1, 84, 84, 1) to (1, 1, 84, 84)

    # Armazene a experiência no buffer de replay do DQN
    model.replay_buffer.add(obs_reshaped, new_obs_reshaped, action, reward, done, info)

    # Atualize o estado atual
    obs = new_obs

    # Treine o modelo periodicamente
    if step > model.learning_starts and step % model.train_freq == 0:
        loss_dict = model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

    # Salve checkpoints
    if step % checkpoint_callback.save_freq == 0:
        checkpoint_callback.on_step()

    # Reinicie o ambiente se o episódio terminar
    if done:
        obs = env.reset()  # Reinicia o ambiente

# Salve o modelo
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")