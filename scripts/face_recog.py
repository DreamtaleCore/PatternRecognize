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

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from utils import DataLoader as dl


# MODEL_PATH = '../models/ORL/alex_net_orl/model'
MODEL_PATH = '../models/PIE/alex_net_orl/model'


def create_simple_network(image_size, n_classes):
    """
    Create the network, using the Simple CovNet backbone
    :param image_size: [width, height, channels]
    :param n_classes: number
    :return: the network
    """
    network = input_data(shape=[None, image_size[0], image_size[1], image_size[2]])

    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 3)
    # network = local_response_normalization(network)
    # network = dropout(network, 0.5)

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 3)
    # network = local_response_normalization(network)
    # network = dropout(network, 0.5)

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 3)
    # network = local_response_normalization(network)
    # network = dropout(network, 0.5)

    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def create_alex_network(image_size, n_classes):
    """
    Create the network, using the AlexNex backbone
    :param image_size: [width, height, channels]
    :param n_classes: number
    :return: the network
    """
    network = input_data(shape=[None, image_size[0], image_size[1], image_size[2]])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def train(net, X, Y):
    model = tflearn.DNN(net, checkpoint_path=MODEL_PATH,
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir=MODEL_PATH+'_tf-dir')
    model.fit(X, Y, n_epoch=1000, validation_set=0.2, shuffle=True,
              show_metric=True, batch_size=512, snapshot_step=200,
              snapshot_epoch=False, run_id='face_recognition')


def predict(network, model_file, images):
    model = tflearn.DNN(network)
    model.load(model_file)
    return model.predict(images)


def run_test():
    X, Y = dl.load_data('ORL', False)

    net = create_alex_network(X[0].shape, Y[0].shape[0])
    model = tflearn.DNN(net)
    model.load(MODEL_PATH + '-2400')

    Y_out = model.predict(X)
    print(len(Y))
    print(Y_out.shape)

    gt = []
    for i in range(len(Y)):
        gt.append(np.argmax(Y[i]))

    y_pred = []
    for i in range(Y_out.shape[0]):
        y_pred.append(np.argmax(Y_out[i, :]))

    right_sum = sum([1 for i in range(len(gt)) if gt[i] == y_pred[i]])
    wrong_idxs = [i for i in range(len(gt)) if gt[i] != y_pred[i]]

    right_rate = float(right_sum) / float(len(gt))
    print('Predict in test dataset, right rate =', right_rate)
    for i in wrong_idxs:
        print('---- in test dataset index:', i, 'predict: ', y_pred[i], 'whereas GT:', gt[i])
    pass


def main():
    X, Y = dl.load_data('PIE')
    print(X[0].shape)
    net = create_alex_network(X[0].shape, Y[0].shape[0])
    train(net, X, Y)


if __name__ == '__main__':
    # run_test()
    main()

