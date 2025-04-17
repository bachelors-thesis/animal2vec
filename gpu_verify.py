import tensorflow as tf

# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)