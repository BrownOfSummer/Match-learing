import tensorflow as tf
import numpy as np
import threading
import time

q = tf.FIFOQueue(2, "int32")
init = q.enqueue_many(([1, 10], ))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)

#
### tf.Coordinator
#
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            coord.request_stop()
        else:
            print("Working on ID: %d\n" % worker_id)
        
        # Sleep 1 second
        print("----------pause----------")
        time.sleep(1)

coord = tf.train.Coordinator()
# Create 5 threds
threads=[ threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5) ]
# Start all the threads
for t in threads: t.start()
# Wait for all the threads exit
coord.join(threads)

#
### tf.QueueRunner()
#
queue = tf.FIFOQueue(100, "float")
# enqueue operation
enqueue_op = queue.enqueue( [tf.random_normal([1])] )
# start 5 threads, and each thread deal with enqueue operation
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)
# Add qr into tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)
# Define dequeue operation
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # Use tf.train.Coordinator to start the threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #get value from queue
    for _ in range(3): print( sess.run(out_tensor)[0] )

    # Use tf.train.Coordinator() to stop threads
    coord.request_stop()
    coord.join(threads)
