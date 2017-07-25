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

INPUT_DATA = '/Users/li_pengju/SomeDownload/Dataset/imgdata/flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 2. Paras of neural network

TRAINING_TFRECORD = './training_features.tfrecord'
TESTING_TFRECORD = './testing_features.tfrecord'
VALIDATION_TFRECORD = './validation_features.tfrecord'


def create_bottleneck(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    """
    Paras:
        sess: Inceptionv3 model graph session
        image_path: Where the image is.
        jpeg_data_tensor: tensor from Inceptionv3 graph
        bottleneck_tensor: tensor from Inceptionv3 graph
    Returns:
        a np array[2048], feature vector
    """
    image_data = gfile.FastGFile(image_path, 'rb').read()
    #bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

    bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)

    return bottleneck_values


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(feature_vector, label, text, image_path):
      """Build an Example proto for an example.
      Args:
        feature_vector: string, a feature vector of an image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'sunflowers'
        image_path: where the image is, for debug
      Returns:
        Example proto
      """
      example = tf.train.Example(features=tf.train.Features(feature={
          'label': _int64_feature(label),
          'text': _bytes_feature(tf.compat.as_bytes(text)),
          'image_path': _bytes_feature(tf.compat.as_bytes(image_path)),
          'feature_vector': _bytes_feature(tf.compat.as_bytes(feature_vector))}))
      return example

def _count_classes():
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    n_classes = 0
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        n_classes += 1
    return n_classes

def create_feature_vector_tfrecord(sess, jpeg_data_tensor, bottleneck_tensor, testing_percentage, validation_percentage):
    """
    Paras:
        sess:
        jpeg_data_tensor: tensor from Inception-v3 graph
        bottleneck_tensor: output tensor from Inception-v3 graph
        testing_percentage: 10
        validation_percentage: 10
    Return:
        n_classes, and 3 tfrecord    
    """
    training_writer = tf.python_io.TFRecordWriter(TRAINING_TFRECORD)
    testing_writer = tf.python_io.TFRecordWriter(TESTING_TFRECORD)
    validation_writer = tf.python_io.TFRecordWriter(VALIDATION_TFRECORD)

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    label = -1
    n_classes = 0
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

        label_name = dir_name.lower() # sunflowers
        label += 1
        n_classes += 1
        for file_name in file_list:
            #base_name = os.path.basename(file_name) # haha.jpg
            feature_vector = create_bottleneck(sess, file_name, jpeg_data_tensor, bottleneck_tensor)
            string_feature_vector = np.array(feature_vector).tostring()
            example = _convert_to_example(string_feature_vector, label=label, text=label_name, image_path=file_name)
            # split the data randomly
            chance = np.random.randint(100)
            if chance < validation_percentage:
                print("label = ",label, "; text = ", label_name, "; ", os.path.basename(file_name), "; ", feature_vector)
                validation_writer.write(example.SerializeToString())
            elif chance < (testing_percentage + validation_percentage):
                print("label = ",label, "; text = ", label_name, "; ", os.path.basename(file_name), "; ", feature_vector)
                testing_writer.write(example.SerializeToString())
            else:
                print("label = ",label, "; text = ", label_name, "; ", os.path.basename(file_name), "; ", feature_vector)
                training_writer.write(example.SerializeToString())
    return n_classes

def main():
    
    # Load the pre-trained Inception-v3 model.
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        n_classes = create_feature_vector_tfrecord(sess, jpeg_data_tensor, bottleneck_tensor, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        print("Totally %d classes !" % n_classes)

if __name__ == '__main__':
    main()
