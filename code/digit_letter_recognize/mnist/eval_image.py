from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib import learn
#from tensorflow.python.platform import gfile

def get_checkpoint_file(MODEL_DIR):
    """
    Input the dir where the trained, return the lasted model path
    """
    #checkpoint_dir = os.path.join(MODEL_DIR,"runs/checkpoints/")
    checkpoint_dir=MODEL_DIR
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

def cnn_query(model_dir, input_data, input_labels=None):
    """
    model_dir: model_dir/checkpoint
    input_data: nx784
    """
    start_eval = time.time()
    # Evaluation
    """Get the checkpoint file"""
    checkpoint_file = get_checkpoint_file(model_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file)) # load model-1350-meta
            saver.restore(sess, checkpoint_file)
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            logits = graph.get_operation_by_name("softmax_linear/logits").outputs[0]
            predictions = tf.argmax(logits, 1, name="predictions")
            #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            predictions = sess.run(predictions,feed_dict={input_x:input_data})
            print(predictions)
            if( input_labels is not None ):
                ground_truth = tf.argmax(input_labels, 1)
                print(sess.run(ground_truth))
    eval_time = time.time() - start_eval
    print("query time:",eval_time)

def cnn_query_image(model_dir, image_path, input_labels=None):
    """
    model_dir: model_dir/checkpoint
    input_data: where the image is
    """
    start_eval = time.time()
    # Evaluation
    """Get the checkpoint file"""
    checkpoint_file = get_checkpoint_file(model_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file)) # load model-1350-meta
            saver.restore(sess, checkpoint_file)
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            logits = graph.get_operation_by_name("softmax_linear/logits").outputs[0]
            predictions = tf.argmax(logits, 1, name="predictions")
            #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            input_data = image2array(sess, image_path, binary_threshold=128, img_size=(28,28), is_resize=True, is_show_img=True)
            # Tensors we want to evaluate
            predictions = sess.run(predictions,feed_dict={input_x:input_data})
            print(predictions)
            if( input_labels is not None ):
                ground_truth = tf.argmax(input_labels, 1)
                print(sess.run(ground_truth))
    eval_time = time.time() - start_eval
    print("query time:",eval_time)

def image2array(sess, image_path, binary_threshold=128, img_size=(28,28), is_resize=True, is_show_img=False ):
    """
    Read a jpeg file, decode, resize, binary and return a ndarray
    Paras:
        image_path: path to image file, should be jpg
        binary_threshold: above set 1, else 0
        image_size: image size after resized, for network input
        is_resize: flag for resize, if for showing, set False
        is_show_img: show image with PILLOW, for confirming
    """
    image_raw_data = tf.gfile.FastGFile(image_path, "r").read()
    src_img_tensor = tf.image.decode_jpeg(image_raw_data)
    if(is_resize):
        resized_tensor = tf.image.resize_images(src_img_tensor, img_size, method=0)
    else:
        resized_tensor = src_img_tensor
    resized = tf.image.rgb_to_grayscale(resized_tensor)
    gray_img = sess.run(resized)
    # turn the gray image to binary with threshold
    #binary_img = gray_img > binary_threshold
    binary_img = gray_img < binary_threshold
    binary_img = binary_img.astype('uint8')
    binary_img.resize( (binary_img.shape[0], binary_img.shape[1]) )
    if(is_show_img):
        img_show = Image.fromarray( np.asarray(binary_img*255, dtype='uint8') )
        img_show.show()
    binary_img.resize((1, binary_img.shape[0] * binary_img.shape[1]))
    return binary_img

MODEL_DIR="/tmp/tensorflow/mnist/logs/fully_connected_feed/"
#get_checkpoint_file(MODEL_DIR)
mnist_data = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",dtype=tf.uint8, one_hot=True)
images = mnist_data.test.images
labels = mnist_data.test.labels
image_batch = images[0:10:1]
label_batch = labels[0:10:1]
#cnn_query(MODEL_DIR, image_batch)
cnn_query(MODEL_DIR, image_batch, label_batch)
image_path="../roi0.jpg"
cnn_query_image(MODEL_DIR, image_path, None)

#mnist_image = image_batch[8]
#mnist_image.resize((28,28))
#show_img = Image.fromarray(np.array(mnist_image, dtype='uint8'))
#show_img.show()
