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

def label2onehot(label_batch, n_classes):
    ground_truthes = []
    for label_index in label_batch:
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truthes.append(list(ground_truth))
    return np.array(ground_truthes)

def get_checkpoint_file(out_dir):
    checkpoint_dir = os.path.join(out_dir,"runs/checkpoints/")
    assert os.path.exists(checkpoint_dir)
    try:
        # If the path unchanged
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        #save_step = checkpoint_file.split('/')[-1].split('-')[-1]
        print('checkpoint_file:',checkpoint_file)
        return checkpoint_file
    except:
        # If the checkpoint path changed
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            latest_checkpoint = ckpt.model_checkpoint_path.split('/')[-1]
            #save_step = latest_checkpoint.split('-')[-1] # 1350
            checkpoint_file = os.path.join(checkpoint_dir, latest_checkpoint) # path + model-1350
            print('checkpoint_file:',checkpoint_file)
            return checkpoint_file
        else:
            print('No checkpoint file found!')
            return None
 
def get_test_data(tfrecord_path):
    labels = []
    feature_vectors = []
    texts = []
    image_pathes = []
    example = tf.train.Example()
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        example.ParseFromString(record)
        f = example.features.feature
        label = f['label'].int64_list.value[0]
        feature_vector = f['feature_vector'].bytes_list.value[0]
        text = f['text'].bytes_list.value[0]
        image_path = f['image_path'].bytes_list.value[0]
        feature_vector = np.fromstring(feature_vector, dtype=np.float32)
        feature_vector = feature_vector.reshape((2048))
        labels.append(label)
        feature_vectors.append(feature_vector)
        texts.append(text)
        image_pathes.append(image_path)
    return feature_vectors, labels, texts, image_pathes


def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [final_tensor_name])
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

