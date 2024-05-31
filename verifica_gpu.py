import tensorflow as tf

# Verificar se a GPU está disponível
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configurar TensorFlow para usar a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar o TensorFlow para não alocar toda a memória da GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow configurado para usar a GPU")
    except RuntimeError as e:
        print(e)

from experiments.train_ppo import train_ppo

if __name__ == '__main__':
    train_ppo('experiments/config.yaml')

