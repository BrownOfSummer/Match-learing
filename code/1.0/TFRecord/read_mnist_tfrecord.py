import tensorflow as tf
import numpy as np
from PIL import Image

def imgshow(image):
    """Show the image from array
    Args:
        image: ndarray
    """
    img_show=Image.fromarray( np.asarray( image, dtype='uint8'))
    img_show.show()
feature_map={"pixels": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64),
        "image_raw": tf.FixedLenFeature([], tf.string)
        }

tfrecord_path = './Records/mnist.tfrecords'
count = 0;
example = tf.train.Example()
sess = tf.Session()
for record in tf.python_io.tf_record_iterator(tfrecord_path):
    count += 1
    if count % 100 == 0:
        print("count = ",count)
        example.ParseFromString(record)
        f = example.features.feature
        #v1 = f['int64 feature'].int64_list.value[0]
        #v2 = f['float feature'].float_list.value[0]
        #v3 = f['bytes feature'].bytes_list.value[0]
        pixels = f['pixels'].int64_list.value[0]
        label = f['label'].int64_list.value[0]
        encoded = f['image_raw'].bytes_list.value[0]
        image_data = tf.decode_raw(encoded,tf.uint8)
        image_data = tf.reshape(image_data,[28, 28])
        image = sess.run(image_data)
        imgshow(image)
        #print(image)
