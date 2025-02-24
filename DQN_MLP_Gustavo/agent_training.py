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
total_timesteps = 30000  # Número de passos de treino
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
    policy=MlpPolicy,
    env=env,
    learning_rate=0.001,  # Não muito alto para evitar aprendizado instável
    buffer_size=50000,  # Reduzido para acelerar aprendizado (100k é bom, mas lento)
    learning_starts=5000,  # Coletar experiências antes de aprender
    batch_size=64,  # Maior batch melhora aprendizado (padrão é 32)
    tau=0.02,  # Atualizações suaves na rede-alvo
    gamma=0.99,  # Mantém aprendizado de longo prazo
    train_freq=4,  # Atualiza a cada 4 passos
    gradient_steps=8,  # Aprende mais por atualização
    target_update_interval=1000,  # Atualiza a rede-alvo a cada 1000 passos
    exploration_fraction=0.3,  # Explora mais no início
    exploration_final_eps=0.05,  # Mantém alguma exploração
    verbose=1,
    tensorboard_log="./tensorboard/dqn_breakout_mlp/"
)

# Treinar o agente
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Salvar o modelo treinado
model.save(save_path)
env.close()

print("Treinamento concluído e modelo salvo!")
