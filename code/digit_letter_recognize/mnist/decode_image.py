import tensorflow as tf
import numpy as np
from PIL import Image

image_raw_data = tf.gfile.FastGFile("../../../imgdata/Lena.jpeg", "r").read()

#with tf.Session() as sess:
#    ## (1) Read image
#    src_img_tensor = tf.image.decode_jpeg(image_raw_data)
#    img_data = sess.run(src_img_tensor)
#    print(img_data.shape)
#    print(type(img_data))
#    print(img_data.dtype)
#    gray_img_tensor = tf.image.rgb_to_grayscale(src_img_tensor)
#    gray_img = sess.run(gray_img_tensor)
#    gray_img.resize((512,512))
#    print(type(gray_img))
#    print(gray_img.dtype)
#    print(gray_img.shape)
#    binary_img = gray_img > 128
#    binary_img = binary_img.astype('uint8')
#    binary_img = binary_img * 255;
#    binary_img.resize((512,512))
#    print(type(binary_img))
#    print(binary_img.dtype)
#    print(binary_img.shape)
#    #img_show = Image.fromarray( np.asarray(img_data, dtype='uint8') )
#    img_show = Image.fromarray( np.asarray(binary_img, dtype='uint8') )
#    img_show.show()

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
    binary_img = gray_img > binary_threshold
    binary_img = binary_img.astype('uint8')
    binary_img.resize( (binary_img.shape[0], binary_img.shape[1]) )
    if(is_show_img):
        img_show = Image.fromarray( np.asarray(binary_img*255, dtype='uint8') )
        img_show.show()
    binary_img.resize((1, binary_img.shape[0] * binary_img.shape[1]))
    return binary_img

with tf.Session() as sess:
    image_array = image2array(sess, "../../../imgdata/Lena.jpeg", img_size=(28,28), is_resize=True, is_show_img=True)
    print(image_array.shape)
    print(image_array)
