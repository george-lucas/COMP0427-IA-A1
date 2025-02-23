import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py


# Inicialize o ambiente
env = gym.make("ALE/Breakout-v5", render_mode="human")
env = AtariWrapper(env)  # Adicione os wrappers do Atari para processamento de frames

# Carregue o modelo treinado
model_path = './models/dqn_breakout.zip'  # Caminho onde o modelo foi salvo
model = DQN.load(model_path)

# Teste o agente treinado
observation, info = env.reset(seed=42)
done = False

while not done:
    env.render()
    
    # Usa o modelo para escolher a ação
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    # Se o episódio terminar ou for truncado, reinicie o ambiente
    if terminated or truncated:
        observation, info = env.reset()

env.close()
