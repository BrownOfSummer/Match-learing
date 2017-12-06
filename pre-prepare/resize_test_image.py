import os
from PIL import Image
scale=0.5
#scale=float(1) / 3.0
PATH_TO_TEST_IMAGES_DIR = '/home/peng_yuxiang/LPJ/data/SceneText/Video/JPEGImages'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'output1_00{}.jpg'.format(i)) for i in range(10, 20)  ]
with open("/home/peng_yuxiang/LPJ/data/SceneText/Video/ImageSets/Main/val.txt",'r') as f:
    test_images = f.readlines()
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, image_name.strip() + '.jpg') for image_name in test_images ]
print TEST_IMAGE_PATHS

for img_name in TEST_IMAGE_PATHS:
    img = Image.open(img_name)
    new_width = int(img.size[0] * scale)
    new_height = int(img.size[1] * scale)
    new_img = img.resize((new_width,new_height), Image.ANTIALIAS)
    new_img.show()
    save_name=img_name.split('/')[-1]
    print save_name
    new_img.save(save_name)
