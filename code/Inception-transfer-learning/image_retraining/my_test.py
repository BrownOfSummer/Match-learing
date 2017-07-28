from __future__ import absulute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import hashlib
import re
import os
import subprocess
from tensorflow.python.util import compat

def run_sys_command(command_str):
    print(command_str)
    stdout = subprocess.PIPE
    stderr = subprocess.PIPE
    p = subprocess.Popen(command_str, shell=True, stdout=stdout, stderr=stderr)
    p.wait()
    out, err = p.communicate()
    print("stdout:",out, "; stderr:",err)

def cp_misclassfy_image_to_dir(misclassify_filename, misclassify_dir):
    """Save misclassify output from retrain.py to file, then cp them to dirs"""
    if not os.path.exists(misclassify_dir):
        os.makedirs(misclassify_dir)
    
    with open(misclassify_filename, "r") as f:
        image_names = f.readlines()
        for image_name in image_names:
            wrong_label = image_name.split(":")[-1].split(" ")[-1]
            image_name = image_name.split(":")[-1].split(" ")[0]
            wrong_label_dir = os.path.join(misclassify_dir, wrong_label)
            if not os.path.exists(wrong_label_dir):
                os.makedirs(wrong_label_dir)
            command_str = " cp {} {}".format(image_name, wrong_label_dir)
            run_sys_command(command_str)

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
def test1():
    file_name="/path/to/just-for-test.jpeg"
    file_name="/path/to/haha_nohash_.jpeg"
    hash_name = re.sub(r'_nohash_.*$', '', file_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
    print("file_name:",file_name,"; hash_name:",hash_name,"; hash_name_hashed:",hash_name_hashed)
    print("hash2num = ",int(hash_name_hashed, 16),"; percentage_hash = ",percentage_hash)

def test2(bottleneck_dir):
    """Test wether the dir can be created auto"""
    bottleneck_path = os.path.join(bottleneck_dir, "inception_v3.txt")
    bottleneck_string = "Just for test !"
    with open(bottleneck_path,'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)

if __name__ == '__main__':
    #test1()
    test2("./tmp_dir/daisy")
