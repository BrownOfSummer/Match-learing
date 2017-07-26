# 1. Path to model and samples
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
N_CLASSES = 5

# 2. trained model and summarys
OUT_DIR = "/tmp/mymodel"

# 3. Inception-v3 model
MODEL_DIR = '../inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

# 4. Percentage of testing and validation while writting tfrecord
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 5. Paras of neural network
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 6. Path to TFRECORD
TRAINING_TFRECORD = './Records/training_features.tfrecord'
TESTING_TFRECORD = './Records/testing_features.tfrecord'
VALIDATION_TFRECORD = './Records/validation_features.tfrecord'

