import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Configurações
env_name = 'ALE/Breakout-v5'
total_timesteps = 100000  # Número de passos de treino
save_path = './models/dqn_breakout'

# Crie o ambiente do Atari 2600
env = make_atari_env(env_name, n_envs=1, seed=42)

# Defina um callback para salvar checkpoints do modelo
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
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
    tensorboard_log="./tensorboard/dqn_breakout/",
    device='cuda'  # Adicione esta linha para usar a GPU
)

# Treine o agente
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Salve o modelo treinado
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")
