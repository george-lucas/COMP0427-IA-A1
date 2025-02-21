import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pymcts import MCTS

# Configurações
env_name = 'ALE/Breakout-v5'
total_timesteps = 100000
save_path = './models/dqn_breakout'

# Crie o ambiente
env = make_atari_env(env_name, n_envs=1, seed=42)

# Callback para salvar checkpoints
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
    tensorboard_log="./tensorboard/dqn_breakout/"
)

# Função para MCTS
def mcts_action(env, state, simulations=100):
    mcts = MCTS(env, state)
    return mcts.search(simulations=simulations)

# Loop de treinamento personalizado
obs = env.reset()
for step in range(total_timesteps):
    # Escolha da ação: alternar entre DQN e MCTS
    if np.random.rand() < 0.01:  # 10% de chance de usar MCTS
        action = mcts_action(env, obs)
    else:
        action, _states = model.predict(obs, deterministic=False)

    # Execute a ação no ambiente
    new_obs, reward, done, info = env.step(action)

    # Armazene a experiência no buffer de replay do DQN
    model.replay_buffer.add(obs, new_obs, action, reward, done)

    # Atualize o estado atual
    obs = new_obs

    # Treine o modelo periodicamente
    if step > model.learning_starts and step % model.train_freq == 0:
        model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

    # Salve checkpoints
    if checkpoint_callback is not None and step % checkpoint_callback.save_freq == 0:
        checkpoint_callback.on_step()

    # Reinicie o ambiente se o episódio terminar
    if done:
        obs = env.reset()

# Salve o modelo
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")