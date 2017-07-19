import tensorflow as tf
import numpy as np
from PIL import Image

np_decode=False
tfrecord_path="../../Records/opt.tfrecords"
count = 0;
example = tf.train.Example()
if(np_decode):

    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        count += 1
        example.ParseFromString(record)
        f = example.features.feature
        height = f['height'].int64_list.value[0]
        width = f['width'].int64_list.value[0]
        image_buffer = f['image_buffer'].bytes_list.value[0]
        text = f['text'].bytes_list.value[0]
        image = np.fromstring(image_buffer, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        img_show=Image.fromarray( np.asarray( image, dtype='uint8'))
        img_show.show()
    print("Totally ",count," records.")

else:
    with tf.Session() as sess:
        for record in tf.python_io.tf_record_iterator(tfrecord_path):
            count += 1
            example.ParseFromString(record)
            f = example.features.feature
            height = f['height'].int64_list.value[0]
            width = f['width'].int64_list.value[0]
            image_buffer = f['image_buffer'].bytes_list.value[0]
            text = f['text'].bytes_list.value[0]
            print(text)
            #image = np.fromstring(image_buffer, dtype=np.uint8)
            image_data = tf.decode_raw(image_buffer, tf.uint8)
            image_data = tf.reshape(image_data,[height,width, 3])
            #print(image_data)
            #print(type(height))
            image = sess.run(image_data) # ndarry
            #print(type(image))
            img_show = Image.fromarray(image)
            img_show.show()
        print("Totally ",count," records.")
