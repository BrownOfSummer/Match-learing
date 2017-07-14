import tensorflow as tf
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("../../imgdata/Lena.jpeg", "r").read()

with tf.Session() as sess:
    ## (1) Read image
    img_data = tf.image.decode_jpeg(image_raw_data)
    #print(img_data.eval())
    #img_data.set_shape([1797, 2673, 3])
    print(img_data.get_shape())
    img_show = Image.fromarray( np.asarray(img_data.eval(), dtype='uint8'))
    img_show.show()
    
    ## (2) Resize image
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    print("Digital type after resize:", resized.dtype, ", Size: ", resized.get_shape())
    uint8_img = np.asarray(resized.eval(), dtype='uint8')
    print("Digital type for image show:", uint8_img.dtype)
    img_show = Image.fromarray(uint8_img)
    #img_show.show()

    ## (3) crop or pad image
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 800, 800)
    img_show = Image.fromarray( np.asarray(croped.eval(), dtype='uint8'))
    #img_show.show()
    img_show = Image.fromarray( np.asarray(padded.eval(), dtype='uint8'))
    #img_show.show()

    ## (4) get 0.5 center area
    central_cropped = tf.image.central_crop(img_data, 0.5)
    img_show = Image.fromarray( np.asarray(central_cropped.eval(), dtype='uint8'))
    #img_show.show()

    ## (5) Rotate the img
    # 上下翻转
    #flipped1 = tf.image.flip_up_down(img_data)
    # 左右翻转
    #flipped2 = tf.image.flip_left_right(img_data)
    
    #对角线翻转
    transposed = tf.image.transpose_image(img_data)
    img_show = Image.fromarray( np.asarray(transposed.eval(), dtype='uint8'))
    #img_show.show()
    
    # 以一定概率上下翻转图片。
    #flipped = tf.image.random_flip_up_down(img_data)
    # 以一定概率左右翻转图片。
    #flipped = tf.image.random_flip_left_right(img_data)

    ## (6) Deal with color

    # 将图片的亮度-0.5。
    #adjusted = tf.image.adjust_brightness(img_data, -0.5)
    
    # 将图片的亮度-0.5
    #adjusted = tf.image.adjust_brightness(img_data, 0.5)
    
    # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
    
    # 将图片的对比度-5
    #adjusted = tf.image.adjust_contrast(img_data, -5)
    
    # 将图片的对比度+5
    adjusted = tf.image.adjust_contrast(img_data, 5)
    
    # 在[lower, upper]的范围随机调整图的对比度。
    #adjusted = tf.image.random_contrast(img_data, lower, upper)

    img_show = Image.fromarray( np.asarray(adjusted.eval(), dtype='uint8'))
    img_show.show()

    ## (7) H and S

    #adjusted = tf.image.adjust_hue(img_data, 0.3)
    #adjusted = tf.image.adjust_hue(img_data, 0.6)
    #adjusted = tf.image.adjust_hue(img_data, 0.9)
    
    # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
    #adjusted = tf.image.random_hue(image, max_delta)
    
    # 将图片的饱和度-5。
    #adjusted = tf.image.adjust_saturation(img_data, -5)
    # 将图片的饱和度+5。
    #adjusted = tf.image.adjust_saturation(img_data, 5)
    # 在[lower, upper]的范围随机调整图的饱和度。
    #adjusted = tf.image.random_saturation(img_data, lower, upper)
    
    # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
    #adjusted = tf.image.per_image_whitening(img_data)
    img_show = Image.fromarray( np.asarray(adjusted.eval(), dtype='uint8'))
    #img_show.show()

    ## (8) Anotation and crop
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=boxes)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0) 
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    
    distorted_image = tf.slice(img_data, begin, size)

    img_show = Image.fromarray( np.asarray(distorted_image.eval(), dtype='uint8'))
    img_show.show()
