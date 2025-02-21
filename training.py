import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

# Inicialize o ambiente
env = gym.make("BreakoutNoFrameskip-v4")  # Usando o ambiente Atari do Gymnasium
env = AtariWrapper(env)  # Adiciona wrappers do Atari para pré-processamento de frames

# Defina o caminho para salvar o modelo treinado e checkpoints
save_path = './models/dqn_breakout'
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path, name_prefix='dqn_breakout_checkpoint')

# Crie o modelo DQN
model = DQN(
    policy="CnnPolicy",  # Use uma CNN para processar os frames do Atari
    env=env,
    learning_rate=0.0001,
    buffer_size=100000,  # Tamanho do buffer de replay
    learning_starts=10000,  # Número de passos antes de começar a treinar
    batch_size=32,  # Tamanho do batch para treinamento
    tau=1.0,  # Taxa de atualização do target network
    gamma=0.99,  # Fator de desconto para recompensas futuras
    train_freq=4,  # Frequência de treinamento (a cada 4 passos)
    gradient_steps=1,  # Número de passos de gradiente por atualização
    target_update_interval=1000,  # Atualizar o target network a cada 1000 passos
    exploration_fraction=0.1,  # Fração do tempo de exploração
    exploration_final_eps=0.01,  # Valor final do epsilon para exploração
    verbose=1,  # Mostrar logs durante o treinamento
    tensorboard_log="./tensorboard/dqn_breakout/"  # Diretório para logs do TensorBoard
)

# Treine o agente
total_timesteps = 100000  # Número total de passos de treinamento
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Salve o modelo treinado
model.save(save_path)

print("Treinamento concluído e modelo salvo!")