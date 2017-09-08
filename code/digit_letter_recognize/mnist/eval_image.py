from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib import learn
#from tensorflow.python.platform import gfile

def get_checkpoint_file(MODEL_DIR):
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

MODEL_DIR="/tmp/tensorflow/mnist/logs/fully_connected_feed/"
#get_checkpoint_file(MODEL_DIR)
mnist_data = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",dtype=tf.uint8, one_hot=True)
images = mnist_data.test.images
labels = mnist_data.test.labels
image_batch = images[0:10:1]
label_batch = labels[0:10:1]
#cnn_query(MODEL_DIR, image_batch)
cnn_query(MODEL_DIR, image_batch, label_batch)
