from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

# 1. Path to model and samples
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
N_CLASSES = 5

MODEL_DIR = '../inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

# feature file dir

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 2. Paras of neural network
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

TRAINING_TFRECORD = './Records/training_features.tfrecord'
TESTING_TFRECORD = './Records/testing_features.tfrecord'
VALIDATION_TFRECORD = './Records/validation_features.tfrecord'

def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def read_and_decode(filename_queue):
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'label': tf.FixedLenFeature([], tf.int64),
          'text': tf.FixedLenFeature([], tf.string),
          'image_path': tf.FixedLenFeature([], tf.string),
          'feature_vector': tf.FixedLenFeature([], tf.string)
      })

    # Convert from a scalar string tensor to a float32 tensor with shape [IMAGE_HEIGHT*IMAGE_WIDTH*3]
    feature_vector = tf.decode_raw(features['feature_vector'], tf.float32)
    feature_vector.set_shape([2048])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    text = tf.cast(features['text'], tf.string)
    image_path = tf.cast(features['image_path'], tf.string)
    return feature_vector, label, text, image_path

def read_inputs(tfrecord_filename, batch_size, num_epochs, shuffle=True):
    """Reads input data num_epochs times.
    Args:
        tfrecord_filename: TFRecord file
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.
        shuffle: whether shuffle the data, shuffle for train, eval do not;
    Returns:
        A tuple (feature_vectors, labels, texts, image_paths), where:
        * feature_vectors is a tf.float32 tensor with shape [2048], maybe do some reshape for train
        * labels is an int32 tensor with shape [batch_size] with the true label, a number in the range [0, NUM_CLASSES).
        * texts is an string tensor with shape [batch_size] human readable image discribe
        * image_paths is a string tensor with shape [batch_size], specify a image, maybe useful for debug
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename queue.
        feature_vector, label, text, image_path = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if(shuffle):
            feature_vectors, sparse_labels, sparse_texts, sparse_image_paths = tf.train.shuffle_batch(
                [feature_vector, label, text, image_path], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000, allow_smaller_final_batch=False)
        else:
            feature_vectors, sparse_labels, sparse_texts, sparse_image_paths = tf.train.batch(
                [feature_vector, label, text, image_path], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size, allow_smaller_final_batch=True)

        return feature_vectors, sparse_labels, sparse_texts, sparse_image_paths


def label2onehot(label_batch, n_classes):
    ground_truthes = []
    for label_index in label_batch:
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truthes.append(list(ground_truth))
    return np.array(ground_truthes)


# 10. main() function
def main():
    n_classes = N_CLASSES
    # Load the pre-trained Inception-v3 model.
    #with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())
    #bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
    #    graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

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

    feature_vectors, labels, _, _ = read_inputs(TRAINING_TFRECORD, batch_size=BATCH, num_epochs=None, shuffle=True)
    validation_vectors, validation_labels, _, _ = read_inputs(VALIDATION_TFRECORD, batch_size=BATCH, num_epochs=None, shuffle=True)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # start training
        try:
            for i in range(STEPS):
                train_bottlenecks, label_ints = sess.run([feature_vectors, labels])
                train_ground_truth = label2onehot(label_ints, n_classes)
                sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % 100 == 0 or i + 1 == STEPS:
                    validation_bottlenecks, label_ints = sess.run([validation_vectors, validation_labels])
                    validation_ground_truth = label2onehot(label_ints, n_classes)
                    validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                    print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))
        except:
            print("Done %d steps." % (i+1))
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    main()
