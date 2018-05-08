"""
This purpose is to convert the `*.mat` files in PIE dataset
to normal images in folder named by its class name
"""
import os
import sys
import time
# For tackling the stupid ROS cv python conflict :-<
try:
    sys.path.remove('/home/ros/ws/ros/pc/devel/lib/python2.7/dist-packages')
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print(e)
import cv2

import numpy as np
import scipy.io as sio

PIE_DATASET_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/PIE_face_dataset'
PIE_OUT_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/PIE_face_dataset_modified'

ORL_DATASET_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112/bmp'
ORL_OUT_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112_modified'


def check_dir(dir_name):
    """If dir is not exist, create it, or do nothing"""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def process_mat_file(filename):
    data_raw = sio.loadmat(filename)

    if data_raw['fea'].shape[0] != data_raw['gnd'].shape[0]:
        print('feature\' number is not match with ground truth\'s number!')
        return None

    check_dir(PIE_OUT_DIR)
    dir_train = os.path.join(PIE_OUT_DIR, 'train')
    dir_test = os.path.join(PIE_OUT_DIR, 'test')
    check_dir(dir_test)
    check_dir(dir_train)

    for i in range(data_raw['fea'].shape[0]):
        if data_raw['isTest'][i] == 1:
            dir_class = os.path.join(dir_test, str(int(data_raw['gnd'][i])))
        else:
            dir_class = os.path.join(dir_train, str(int(data_raw['gnd'][i])))
        check_dir(dir_class)

        img_size = int(np.sqrt(data_raw['fea'][i].shape[0]))
        img = data_raw['fea'][i].reshape(img_size, img_size)

        img_name = str(time.time()) + '.jpg'

        cv2.imshow('vis', img)
        cv2.imwrite(os.path.join(dir_class, img_name), img)
        cv2.waitKey(5)


def split_dataset():
    check_dir(ORL_OUT_DIR)
    dir_train = os.path.join(ORL_OUT_DIR, 'train')
    dir_test = os.path.join(ORL_OUT_DIR, 'test')
    check_dir(dir_train)
    check_dir(dir_test)

    for root, dirs, class_files in os.walk(ORL_DATASET_DIR):
        n_files = len(class_files)
        class_idx = 0
        for class_name in dirs:
            class_idx = class_idx + 1
            print('## %d/%d' % (class_idx, n_files), '| processing class:', class_name, '.' * 3)
            dir_class = os.path.join(ORL_DATASET_DIR, class_name)

            for _, _, img_names in os.walk(dir_class):
                img_idx = 0
                img_names.sort()
                for img_name in img_names:
                    # avoid bad files
                    if img_name == 'Thumbs.db':
                        continue

                    dir_img = os.path.join(dir_class, img_name)
                    img = cv2.imread(dir_img)
                    if img_idx == 4 or img_idx == 8:
                        dir_img_out = os.path.join(dir_test, class_name)
                    else:
                        dir_img_out = os.path.join(dir_train, class_name)

                    check_dir(dir_img_out)

                    cv2.imshow('vis', img)
                    cv2.imwrite(os.path.join(dir_img_out, img_name), img)
                    cv2.waitKey(5)

                    img_idx = img_idx + 1


def main_1():
    # first of all, walk dataset dir to get all `*.mat` files
    t_begin = time.time()
    for root, dirs, files in os.walk(PIE_DATASET_DIR, topdown=True):
        n_files = len(files)
        idx = 0
        for name in files:
            idx = idx + 1
            print('## %d/%d' % (idx, n_files), '| processing file:', name, '.'*3)
            process_mat_file(os.path.join(PIE_DATASET_DIR, name))

    print('done!')
    print('Elapsed time: ', time.time() - t_begin, 's.')


def main_2():
    t_begin = time.time()
    split_dataset()

    print('done!')
    print('Elapsed time: ', time.time() - t_begin, 's.')


if __name__ == '__main__':
    main_2()
