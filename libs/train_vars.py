from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.python.client import device_lib
import os


hostname = os.uname()[1]


#https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html

# nthreads=8
# tf.config.threading.set_inter_op_parallelism_threads(nthreads) 
# tf.config.threading.set_intra_op_parallelism_threads(nthreads)
# tf.config.set_soft_device_placement(True)

# os.environ["OMP_NUM_THREADS"] = str(nthreads)



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    
def tf_device_test():
    options = get_available_gpus()
    if hostname == "simon-pc":
        return "/GPU:0"
    
    if "/device:XLA_GPU:0" in options:
        return "/XLA_GPU:0"
    elif "/device:GPU:0" in options:
        return "/GPU:0"
    elif "/device:XLA_CPU:0" in options:
        return "/XLA_CPU:0"
    else: 
        return "/CPU:0"

tf.config.optimizer.set_jit(True)
chosen_device = tf_device_test()
tf.device(chosen_device)
print(f"Using device: {chosen_device}")


CELL = LSTM

LOSS = "mse"
# LOSS = "mae"
OPTIMIZER = "adam"
# OPTIMIZER = "rmsprop"
BATCH_SIZE = 2**14
EPOCHS = 10000000
LOOP_NUM = False

if chosen_device in ["/XLA_CPU:0", "/CPU:0"]:
    BATCH_SIZE = 2048
    EPOCHS = 2000
    LOOP_NUM = 1
