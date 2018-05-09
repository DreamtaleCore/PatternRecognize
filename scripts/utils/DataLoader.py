import os
import sys
# For tackling the stupid ROS cv python conflict :-<
try:
    sys.path.remove('/home/ros/ws/ros/pc/devel/lib/python2.7/dist-packages')
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print(e)
import cv2
import numpy as np

PIE_DATASET_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/PIE_face_dataset_modified/'

ORL_DATASET_DIR = '/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112_modified'


def load_data(datafile, is_train=True):
    """
    Load face dataset from dist
    :param datafile: 'PIE' or 'ORL'
    :return:
    """
    labels = []
    images = []

    if datafile == 'PIE':
        dataset_dir = PIE_DATASET_DIR
    elif datafile == 'ORL':
        dataset_dir = ORL_DATASET_DIR
    else:
        dataset_dir = datafile

    if is_train:
        dataset_dir = os.path.join(dataset_dir, 'train')
    else:
        dataset_dir = os.path.join(dataset_dir, 'test')

    for root, class_dirs, files in os.walk(dataset_dir):
        class_dirs.sort()
        n_classes = len(class_dirs)
        for class_name in class_dirs:
            if class_name[0].isalpha():
                class_id = int(class_name[1:])
            else:
                class_id = int(class_name)

            dir_class = os.path.join(dataset_dir, class_name)
            for _, _, img_names in os.walk(dir_class):
                img_names.sort()
                for img_name in img_names:
                    # avoid bad files
                    if img_name == 'Thumbs.db':
                        continue

                    img_dir = os.path.join(dir_class, img_name)
                    image = cv2.imread(img_dir)

                    images.append(image)
                    label = np.zeros(n_classes)
                    label[class_id - 1] = 1
                    labels.append(label)

    return images, labels


