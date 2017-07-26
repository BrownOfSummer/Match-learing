from config import *
from utils import *

def main():
    n_classes = N_CLASSES
    # Load the pre-trained Inception-v3 model.
    #with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())
    #bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
    #    graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # define new input of network
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    
    # define a fully connect layer for classfing
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        
    # define cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    
    # calc right percentage of a batch or all testing vectors
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="evaluation_step")

    feature_vectors, labels, _, _ = read_inputs(TRAINING_TFRECORD, batch_size=BATCH, num_epochs=None, shuffle=True)
    validation_vectors, validation_labels, _, _ = read_inputs(VALIDATION_TFRECORD, batch_size=BATCH, num_epochs=None, shuffle=True)
    with tf.Session() as sess:

        loss_summary = tf.summary.scalar("loss", cross_entropy_mean) # loss
        acc_summary = tf.summary.scalar("accuracy", evaluation_step) # Validation accuracy
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(OUT_DIR, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        checkpoint_dir = os.path.abspath(os.path.join(OUT_DIR, "runs", "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # start training
        try:
            for i in range(STEPS):
                train_bottlenecks, label_ints = sess.run([feature_vectors, labels])
                train_ground_truth = label2onehot(label_ints, n_classes)
                sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % 100 == 0 or i + 1 == STEPS:
                    validation_bottlenecks, label_ints = sess.run([validation_vectors, validation_labels])
                    validation_ground_truth = label2onehot(label_ints, n_classes)
                    validation_accuracy, summaries = sess.run([evaluation_step, train_summary_op], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                    print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))
                    path = saver.save(sess, checkpoint_prefix, global_step=i)
                    print("Saved model checkpoint to {}\n".format(path))
                    train_summary_writer.add_summary(summaries, i)
        except:
            print("Done %d steps." % (i+1))
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    main()
