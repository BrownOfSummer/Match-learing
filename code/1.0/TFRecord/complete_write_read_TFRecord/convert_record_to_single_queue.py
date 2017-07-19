from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from PIL import Image

IMAGE_HEIGHT = 388
IMAGE_WIDTH = 584
tfrecord_filename="../../Records/opt.tfrecords"
def read_and_decode(filename_queue):
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'text': tf.FixedLenFeature([], tf.string),
          'image_buffer': tf.FixedLenFeature([], tf.string)
      })

    # Convert the image data from string back to the numbers
    # Convert from a scalar string tensor to a uint8 tensor with shape [IMAGE_HEIGHT*IMAGE_WIDTH*3]
    image = tf.decode_raw(features['image_buffer'], tf.uint8)
    #image.set_shape([mnist.IMAGE_PIXELS])
    # reshape for showing image
    # Reshape image data into the original shape
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # OPTIONAL: Could reshape into a IMAGE_HEIGHT*IMAGE_WIDTH*3 image and apply distortions here
    # Since we are not applying any distortions in this example, and the training step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    text = tf.cast(features['text'], tf.string)
    return image, label, text

def main():
    num_epochs = 2 #read twice
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=num_epochs)
    image, label, text = read_and_decode(filename_queue)
    print("image_type:",type(image)," label_type:",type(label)," text_type:",type(text)) # both tensor
    # Initialize all global and local variables, important
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        try:
          while not coord.should_stop():
            # Run one step of getting image and label, runing result should not same as op:
            # that is, img should not be image, etc
            img, lbl, title = sess.run([image, label, text]) # get ndarray
            # show the image
            img_show = Image.fromarray(img)
            img_show.show()
            print(title)
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
    main()
