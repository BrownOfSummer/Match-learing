from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from PIL import Image
def imgshow(image):
    """Show the image from array
    Args:
        image: ndarray
    """
    img_show=Image.fromarray( np.asarray( image, dtype='uint8'))
    img_show.show()

def _imgshow(image_data, coder):
    """Show the image_data decode from jpeg or png.
    Args:
        image_data: decoded data from jpeg or png.
        coder: Helper class that provides TensorFlow image coding utilities.
    """
    image = coder.decode_jpeg(image_data)
    img_show = Image.fromarray( np.asarray(image, dtype='uint8'))
    img_show.show()

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'channels': _int64_feature(channels),
      'label': _int64_feature(label),
      'text': _bytes_feature(tf.compat.as_bytes(text)),
      'format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer.tostring()))}))
  return example

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename

def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  #return image_data, height, width
  return image, height, width

def main():
    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()
    filename_list = ["/Users/li_pengju/SomeDownload/Dataset/imgdata/Fruits.jpg",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/Airplane.jpg",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/Baboon.jpg"]

    """Write to TFRecords"""
    record_name = "./Records/test.tfrecords"
    writer = tf.python_io.TFRecordWriter(record_name)
    for filename in filename_list:
        image_data, height, width = _process_image(filename, coder)
        imgshow(image_data)
        #example = _convert_to_example(filename, image_data, 1, "Fruits", height, width)
        example = _convert_to_example(filename, image_data, 1, filename, height, width)
        writer.write(example.SerializeToString())
    writer.close()
    print("TFRecords saved in ", record_name)

if __name__ == '__main__':
    main()
