from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, text, height, width):
      """Build an Example proto for an example.
      Args:
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
      Returns:
        Example proto
      """
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(height),
          'width': _int64_feature(width),
          'label': _int64_feature(label),
          'text': _bytes_feature(tf.compat.as_bytes(text)),
          'image_buffer': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
      return example

def _process_image(filename):
      """Process a single image file.
      Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
      Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
      """
      # Read the image file.
      image = np.array(Image.open(filename))
      # May be add another process here: resize, convert, ...
      # Check that image converted to RGB
      assert len(image.shape) == 3
      height = image.shape[0]
      width = image.shape[1]
      assert image.shape[2] == 3
      image_buffer = image.tostring()
      return image_buffer, height, width

def main():
    filename_list = ["/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame07.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame08.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame09.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame10.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame11.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame12.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame13.png",
            "/Users/li_pengju/SomeDownload/Dataset/imgdata/optical_flow_image/eval-data/Army/frame14.png",
            ]
    record_name = "../../Records/opt.tfrecords"
    writer = tf.python_io.TFRecordWriter(record_name)
    for filename in filename_list:
        image_buffer, height, width = _process_image(filename)
        example = _convert_to_example(image_buffer, 1, "Army", height, width)
        writer.write(example.SerializeToString())
        # show the image
        print("height=",height, "; width=",width)
        image = np.fromstring(image_buffer, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        img_show=Image.fromarray( np.asarray( image, dtype='uint8'))
        img_show.show()

    writer.close()
    print("TFRecords saved in ", record_name)

if __name__ == '__main__':
    main()
