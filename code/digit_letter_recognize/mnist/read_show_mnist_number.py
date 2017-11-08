from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
# define at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
def imgshow(image):
    """Show the image from array
    Args:
    image: ndarray
    """
    img_show=Image.fromarray( np.asarray( image, dtype='uint8' ) )
    img_show.show()
# Read mnist data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",dtype=tf.uint8, one_hot=True)
images = mnist.validation.images #test, train, validation
labels = mnist.validation.labels #test, train, validation
pixels = images.shape[1]
numbers = images.shape[0]
num_examples = mnist.train.num_examples

#print("images[0] = ",images[0])     # image data
print(type(images[0]));
#print("images[0].tostring",images[0].tostring())
print("labels[0] = ",type(labels[0]),labels[0])     # on-hot
print("pixels(cols) = ",pixels)           #784
print("numbers(rows) = ",numbers)
print("images.shape(rows, cols) = ",images.shape) #(55000, 784)
print("num_examples(totol images) = ",num_examples) #55000
array_resize = images[1000]
array_resize.resize((28,28))
print(array_resize.shape)
imgshow(array_resize)
image_vector = images[0:10:1]
print(image_vector.shape)
