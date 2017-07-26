from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 1. Path to model and samples
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '../inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

# feature file dir
CACHE_DIR = './bottleneck'
INPUT_DATA = '/Users/li_pengju/SomeDownload/Dataset/imgdata/flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 2. Paras of neural network
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 3. List all the images, and split them according to training, testing and validation 
def create_image_lists(testing_percentage, validation_percentage):
    """Paras:
    testing_percentage: testing images percentage of each class;
    validation_percentage: validation images percentage of each class;
    INPUT_DATA: where the images is, /path/flower_photos
    return:
        A dict of all images' path and name
        result = 
        {
            "roses":
            {
                "dir":roses
                "training":[1.jpg, 8.jpeg, lala.JPG, haha.JPEG, ...]
                "testing":[4.jpg, 5.jpeg, 101.JPG, aha.JPEG, ...]
                "validation":[some_image_names.jpg, ...]
            } 
            
            label_name3:
            {
                "dir":class_dir
                "training":[]
                ...
            }
            ...

        }

    """
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()
        
        # initialize
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # split the data randomly
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result

# 4. Get the path of a image using class_name, category of training or testing and the index of the image list;
def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    Paras:
    image_lists: The dict get from create_image_lists
    image_dir: /path/flower_photos
    label_name: "roses", "sunflowers", ...
    category: training, testing or validation
    index: an int number, index % len([training , testing or validation list]) to void exceed
    Return:
    /path/flower_photos/sunflowers/aha.jpg, or another image path ...
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 5. Define the path to save the Inception-v3 feature vector.
def get_bottleneck_path(image_lists, label_name, index, category):
    """
    Get a txt path to save featrue, simply add ".txt" after image path
    Paras:
    CACHE_DIR: where to save the .txt feature file.
    Return:
        /path/flower_photos/sunflowers/aha.jpg.txt
    """
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# 6. Use the loaded Inception-v3 session to process an image, then get the feature vector of the image.
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """
    Paras:
        sess:
        image_data: gfile.FastGFile read image data;
        image_data_tensor: placeholder form Inception-v3 graph.
        bottleneck_tensor: output tensor of bottleneck layer
    Return:
        feature vector of the image:[2048]
    """

    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 7. Attemp to find the calculated feature vector, if none, calculate the feature and the save it.
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    """
    Paras:
        sess:
        image_lists: image file and path dict.
        label_name: "roses", "sunflowers", ...
        index: an int number to index an image in list.
        category: "training", "testing" or "validation"
        jpeg_data_tensor: placeholder tensor from Inception-v3 graph
        bottleneck_tensor: output tensor of bottleneck layer
        CACHE_DIR: where to write the .txt file.
    Return:
        feature vector of an image:[2048]
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):

        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)

        image_data = gfile.FastGFile(image_path, 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 8. Randomly get a batch of images as traing data.
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    """
    Paras:
        sess:
        n_classes: 5 classes
        image_lists: image file and path dict.
        how_many: batch size
        category: "training", "testing" or "validation"
        jpeg_data_tensor: placeholder tensor from Inception-v3 graph
        bottleneck_tensor: output tensor of bottleneck layer
    Return:
        feature vector batch.
        ground_truth one-hot vector batch.
    """
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

# 9. Get all the testing feature vectors
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    """
    Return:
        feature vector batch of all testing.
        the according ground_truth one-hot vector batch.
    """
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 10. main() function
def main():
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # Load the pre-trained Inception-v3 model.
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # define new input of network
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    
    # define a fully connect layer for classfing
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        
    # define cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    
    # calc right percentage of a batch or all testing vectors
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # start training
        for i in range(STEPS):
 
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                    (i, BATCH, validation_accuracy * 100))
            
        # calc the right percentage of testing data
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()
