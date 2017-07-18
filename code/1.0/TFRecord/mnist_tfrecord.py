import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#
### 1 input -> TFRecord
#
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def  _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# Read mnist data
mnist = input_data.read_data_sets("/Users/li_pengju/SomeDownload/Dataset/MNIST_data",dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

#print("images[0] = ",images[0])     # image data
#print("images[0].tostring",images[0].tostring())
#print("labels[0] = ",labels[0])     # on-hot
#print("pixels = ",pixels)           #784
#print("images.shape = ",images.shape) #(55000, 784)
#print("num_examples = ",num_examples) #55000

# Output TFRecord
filename="./Records/mnist.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(1000):
    image_raw = images[index].tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
print("TFRecords saved in ",filename)

# Read TFRecord
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["Records/mnist.tfrecords"])
_, serialized_example = reader.read(filename_queue)

# analyze
features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([], tf.string),
            'pixels':tf.FixedLenFeature([], tf.int64),
            'label':tf.FixedLenFeature([], tf.int64)
        })

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# multi threads deal with input data
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(5):
    image, label, pixel = sess.run([images, labels, pixels])
    print("label = ",label)
    print("pixels = ",pixels)
