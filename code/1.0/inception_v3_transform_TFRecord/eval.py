from config import *
from utils import *
"""Get the checkpoint file"""
checkpoint_file = get_checkpoint_file(OUT_DIR)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file)) # load model-1350-meta
        saver.restore(sess, checkpoint_file)
        # Get the placeholders from the graph by name
        bottleneck_input = graph.get_operation_by_name("BottleneckInputPlaceholder").outputs[0]
        ground_truth_input = graph.get_operation_by_name("GroundTruthInput").outputs[0]
        # Get tensors from graph by name
        evaluation_step = graph.get_operation_by_name("evaluation/evaluation_step").outputs[0]

        test_bottlenecks, labels, texts, image_pathes = get_test_data(TESTING_TFRECORD)
        test_ground_truth = label2onehot(labels, N_CLASSES)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
