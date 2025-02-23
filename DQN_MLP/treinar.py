import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Configurações
env_name = 'ALE/Breakout-ram-v5'  # "-ram" para observações de memória
total_timesteps = 500000
save_path = './models/dqn_breakout_ram'

# Criar ambiente com observações RAM
def make_env():
    env = gym.make(env_name, render_mode='rgb_array')
    return env

env = DummyVecEnv([make_env])  # Ambiente vetorizado

# Normalizar observações RAM (0-255 → 0-1)
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Callback para salvar checkpoints
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
                                         name_prefix='dqn_ram')

# Modelo DQN com política MLP para dados RAM
model = DQN(
        policy="MlpPolicy", # Usar MLP em vez de CNN
        env=env,
        learning_rate=0.0001,
        buffer_size=100000,
        batch_size=32,
        learning_starts=10000,
        tau = 0.1, # NOTE: acredito que não funciona
        gamma = 0.99, # NOTE: acredito que não funciona
        train_freq=4,
        target_update_interval=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,

        policy_kwargs=dict(
            net_arch=[256, 256]  # Arquitetura da rede neural
            ),
        verbose=1,
        tensorboard_log="./tensorboard/dqn_breakout_ram/"
        )

# Treinar
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save(save_path)
env.save(save_path + "_vecnormalize.pkl")  # Salva as estatísticas
env.close()

