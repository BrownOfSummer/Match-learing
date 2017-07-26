from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from config import *
from utils import *

def run(tfrecord_filename):
    """Get batches of image, label and texts; then, show them"""
    num_epochs = 1
    batch_size = 100
    # shuffle = False will allow_smaller_final_batch
    feature_vectors, labels, texts, image_paths = read_inputs(tfrecord_filename, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
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
