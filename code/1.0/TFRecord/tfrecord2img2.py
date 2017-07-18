import tensorflow as tf
import numpy as np
from img2tfrecord2 import ImageCoder
from img2tfrecord2 import _imgshow
from PIL import Image

#features = {"height": tf.FixedLenFeature([], tf.int64),
#        "width": tf.FixedLenFeature([], tf.int64),
#        "colorspace": tf.FixedLenFeature([], tf.string),
#        "channels": tf.FixedLenFeature([], tf.int64),
#        "label": tf.FixedLenFeature([], tf.int64),
#        "text": tf.FixedLenFeature([], tf.string),
#        "format": tf.FixedLenFeature([], tf.string),
#        "filename": tf.FixedLenFeature([], tf.string),
#        "encoded": tf.FixedLenFeature([], tf.string) }

tfrecord_path = './Records/test.tfrecords'
coder = ImageCoder()
count = 0;
example = tf.train.Example()
sess = tf.Session()
for record in tf.python_io.tf_record_iterator(tfrecord_path):
    count += 1
    print("count = ",count)
    example.ParseFromString(record)
    f = example.features.feature
    #v1 = f['int64 feature'].int64_list.value[0]
    #v2 = f['float feature'].float_list.value[0]
    #v3 = f['bytes feature'].bytes_list.value[0]
    height = f['height'].int64_list.value[0]
    width = f['width'].int64_list.value[0]
    encoded = f['encoded'].bytes_list.value[0]
    _imgshow(encoded, coder)
    image = coder.decode_jpeg(encoded)
    #image_data = tf.reshape(image_data,[height, width, 3])
    #image = sess.run(image_data)
    #height = tf.cast(height, tf.int64)
    #width = tf.cast(width, tf.int64)
    print(type(height))
    print(type(image))
