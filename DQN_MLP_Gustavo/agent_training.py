import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Configurações
env_name = 'ALE/Breakout-v5'
total_timesteps = 5000  # Número de passos de treino (ajustado para 1000)
save_path = './models/dqn_breakout_mlp'

# Criar o ambiente do Atari 2600
env = gym.make(env_name)
env = Monitor(env, "./logs")  # Defina o diretório para logs
env = AtariWrapper(env)  # Usando o AtariWrapper

# Callback para salvar checkpoints do modelo
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,
                                         name_prefix='dqn_breakout_mlp_checkpoint')

# Criar o modelo DQN com uma rede MLP
model = DQN(
    policy=MlpPolicy,  # Usando uma rede MLP
    env=env,
    learning_rate=0.8001,
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
    tensorboard_log="./tensorboard/dqn_breakout_mlp/"
)

# Treinar o agente
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Salvar o modelo treinado
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")
