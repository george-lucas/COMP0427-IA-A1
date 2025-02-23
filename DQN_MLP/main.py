import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Configurações
env_name = 'ALE/Breakout-ram-v5'
model_path = './models/dqn_breakout_ram'  # Caminho do modelo salvo
vecnorm_path = './models/dqn_breakout_ram_vecnormalize.pkl'  # Caminho dos stats de normalização

# 1. Criar ambiente (com mesmo setup do treino)
def make_env():
    return gym.make(env_name, render_mode='human')  # Modo de renderização "human"

env = DummyVecEnv([make_env])

# 2. Carregar normalização (se usou VecNormalize)
env = VecNormalize.load(vecnorm_path, env)
env.training = False  # Desativa atualização das estatísticas
env.norm_reward = False  # Mantém as recompensas originais

# 3. Carregar modelo
model = DQN.load(model_path, env=env)

# 4. Executar demonstração
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)  # Ação determinística
    obs, reward, done, _ = env.step(action)

    # Renderizar o jogo
    env.render()

    if done:
        obs = env.reset()

