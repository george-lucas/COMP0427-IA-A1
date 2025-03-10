import gymnasium as gym
import numpy as np
import ale_py
from force_fire import ForceFireEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Configurações
env_name = 'ALE/Breakout-ram-v5'  # "-ram" para observações de memória
total_timesteps = 500_000
save_path = './models/dqn_breakout_ram'


# Criar ambiente com observações RAM
def make_env():
    env = ForceFireEnv(
            gym.make(env_name, render_mode='rgb_array'))
    env = Monitor(env, "./logs")  # Defina o diretório para logs

    return env

env = DummyVecEnv([make_env])  # Ambiente vetorizado
# env = SubprocVecEnv(env)  # Ambiente vetorizado

# Normalizar observações RAM (0-255 → 0-1)
env = VecNormalize(env, norm_obs=False, norm_reward=False)


# Callback para salvar checkpoints
# checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
#                                          name_prefix='dqn_ram')

# Modelo DQN com política MLP para dados RAM
arch = [256, 256, 256]
model_name = "_".join(str(x) for x in [*arch, total_timesteps])
model = DQN(
        policy="MlpPolicy", # Usar MLP em vez de CNN
        env=env,
        learning_rate=0.0001,
        buffer_size=100000,
        batch_size=32,
        learning_starts=10000,
        tau = 0.1,
        gamma = 0.99,
        train_freq=4,
        target_update_interval=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,

        policy_kwargs=dict(
            net_arch=arch  # Arquitetura da rede neural
            ),
        verbose=1,
        tensorboard_log=f"./tensorboard/dqn_breakout_ram_{model_name}/"
        )

def gerar_png_modelo(model):
    import torch as torch
    from torchviz import make_dot
    fake_input = torch.randn(1, 128)  # Entrada fictícia (128 bytes da RAM)
    q_values = model.policy.q_net(fake_input)  # Passa pela rede
    dot = make_dot(q_values, params=dict(model.policy.q_net.named_parameters()))
    dot.format = "png"
    dot.render("model_architecture")  # Salva como "dqn_architecture.png"

gerar_png_modelo(model)

# Treinar
model.learn(total_timesteps=total_timesteps)
model.save(save_path)
env.save(save_path + "_vecnormalize.pkl")  # Salva as estatísticas
env.close()

