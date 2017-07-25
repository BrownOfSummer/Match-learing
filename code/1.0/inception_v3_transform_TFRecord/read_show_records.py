from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

TRAINING_TFRECORD = './Records/training_features.tfrecord'
TESTING_TFRECORD = './Records/testing_features.tfrecord'
VALIDATION_TFRECORD = './Records/validation_features.tfrecord'

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

    # Convert the image data from string back to the numbers
    # Convert from a scalar string tensor to a uint8 tensor with shape [IMAGE_HEIGHT*IMAGE_WIDTH*3]
    #image = tf.decode_raw(features['image_buffer'], tf.uint8)
    feature_vector = tf.decode_raw(features['feature_vector'], tf.float32)
    feature_vector.set_shape([2048])
    #image.set_shape([mnist.IMAGE_PIXELS])
    # reshape for showing image
    # Reshape image data into the original shape
    #image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # OPTIONAL: Could reshape into a IMAGE_HEIGHT*IMAGE_WIDTH*3 image and apply distortions here
    # Since we are not applying any distortions in this example, and the training step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

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
    A tuple (images, labels, texts), where:
    * images is a uint8 tensor with shape [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], maybe do some reshape for train
    * labels is an int32 tensor with shape [batch_size] with the true label, a number in the range [0, NUM_CLASSES).
    * texts is an string tensor with shape [batch_size] human readable image discribe
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
                min_after_dequeue=1000)
        else:
            feature_vectors, sparse_labels, sparse_texts, sparse_image_paths = tf.train.batch(
                [feature_vector, label, text, image_path], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size)

        return feature_vectors, sparse_labels, sparse_texts, sparse_image_paths

def run(tfrecord_filename):
    """Get batches of image, label and texts; then, show them"""
    num_epochs = 1
    batch_size = 100
    feature_vectors, labels, texts, image_paths = read_inputs(tfrecord_filename, batch_size=batch_size, num_epochs=num_epochs, shuffle=False)
    #print("feature_vectors_type:",type(feature_vectors)," labels_type:",type(labels)," texts_type:",type(texts)) # both tensor
    # The op for initializing the variables, important
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
          while not coord.should_stop():
            # Run one step of getting image and label, runing result should not same as op:
            # that is, img should not be image, etc
            feature_vector_batch, label_batch, text_batch, image_path_batch = sess.run([feature_vectors, labels, texts, image_paths]) # get ndarray
            # show the image
            for index in range(batch_size):
                print("label = ",label_batch[index], "; text = ", text_batch[index], "; ", image_path_batch[index], "; ", feature_vector_batch[index])
            step += 1
        except tf.errors.OutOfRangeError:
            print('Done reading for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
    print("step = ",step);

if __name__ == '__main__':
    run(TRAINING_TFRECORD)
