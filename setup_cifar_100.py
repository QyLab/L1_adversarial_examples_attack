## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.models import load_model

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels


def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32 * 32 * 3 + 1 + 1
    labels_c = []
    labels_f = []
    images = []
    for i in range(50000):
        arr = np.fromstring(f[i * size:(i + 1) * size], dtype=np.uint8)
        lab_coarse = np.identity(20)[arr[0]]
        lab_fine = np.identity(100)[arr[1]]
        img = arr[2:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels_c.append(lab_coarse)
        labels_f.append(lab_fine)
        images.append((img / 255) - .5)
    return np.array(images), np.array(labels_c), np.array(labels_f)


def load_batch_test(fpath):
    f = open(fpath, "rb").read()
    size = 32 * 32 * 3 + 1 + 1
    labels_c = []
    labels_f = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i * size:(i + 1) * size], dtype=np.uint8)
        lab_coarse = np.identity(20)[arr[0]]
        lab_fine = np.identity(100)[arr[1]]
        img = arr[2:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels_c.append(lab_coarse)
        labels_f.append(lab_fine)
        images.append((img / 255) - .5)
    return np.array(images), np.array(labels_c), np.array(labels_f)


class CIFAR_100:
    def __init__(self):
        train_data = []
        train_labels = []

        data_set_path = "../cifar_100/cifar-100-binary/"
        r, s1, s2 = load_batch(data_set_path + "train.bin")
        train_data.extend(r)
        train_labels.extend(s2)

        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels)

        self.test_data, self.test_labels_c, self.test_labels = load_batch_test(data_set_path + "test.bin")

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class CIFAR_100_Model:
    def __init__(self, restore, image_size=32, cnn=1, channels=3):
        self.num_channels = channels
        if cnn:
            self.image_size = image_size
        else:
            self.image_size = image_size * image_size * self.num_channels

        self.root_name = restore
        reader = tf.train.NewCheckpointReader(self.root_name + "checkpoint_dir/model.ckpt")
        self.data_dir = self.root_name + 'data/'
        weight_size = np.loadtxt(self.data_dir + 'weight_size.txt', dtype=int)
        fc_node_number = np.loadtxt(self.data_dir + 'fc_node_number.txt', dtype=int)
        self.num_labels = fc_node_number[len(fc_node_number) - 1]
        model = Sequential()

        if cnn:
            kernel_size = np.loadtxt(self.data_dir + 'kernel_size.txt', dtype=int)
            kernel_number = np.loadtxt(self.data_dir + 'kernel_number.txt', dtype=int)
            pool_size = np.loadtxt(self.data_dir + 'pool_size.txt', dtype=int)
            padding = np.loadtxt(self.data_dir + 'padding.txt', dtype=int)
            features = np.loadtxt(self.data_dir + 'features.txt', dtype=int)

            # 输入层连接的
            model.add(Conv2D(kernel_number[0], (kernel_size[0], kernel_size[0]),
                             input_shape=(self.image_size, self.image_size, self.num_channels),
                             kernel_initializer=keras.initializers.Constant(reader.get_tensor('W1')),
                             bias_initializer=keras.initializers.Constant(reader.get_tensor('b1'))))
            model.add(Activation('relu'))
            for i in range(1, weight_size - 1):  # 最后一层没有relu
                if features[i] == 1:
                    model.add(Conv2D(kernel_number[i], (kernel_size[i], kernel_size[i]),
                                     kernel_initializer=keras.initializers.Constant(
                                         reader.get_tensor('W' + str(i + 1))),
                                     bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(i + 1)))))
                    model.add(Activation('relu'))
                elif features[i] == 3:
                    model.add(MaxPooling2D(pool_size=(pool_size[i], pool_size[i])))
                elif features[i] == 2:
                    model.add(AveragePooling2D(pool_size=(pool_size[i], pool_size[i])))
                elif features[i] == 0:  # 全连接
                    if features[i - 1] != 0:  # 拉平
                        model.add(Flatten())
                    model.add(Dense(fc_node_number[i],
                                    kernel_initializer=keras.initializers.Constant(reader.get_tensor('W' + str(i + 1))),
                                    bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(i + 1)))))
                    model.add(Activation('relu'))
            model.add(Dense(fc_node_number[weight_size - 1],
                            kernel_initializer=keras.initializers.Constant(reader.get_tensor('W' + str(weight_size))),
                            bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(weight_size)))))
        else:
            # 输入层连接的
            model.add(Dense(fc_node_number[0], input_dim=self.image_size,
                            kernel_initializer=keras.initializers.Constant(reader.get_tensor('W' + str(0 + 1))),
                            bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(0 + 1)))))
            model.add(Activation('relu'))
            for i in range(1, weight_size - 1):  # 最后一层没有relu
                model.add(Dense(fc_node_number[i],
                                kernel_initializer=keras.initializers.Constant(reader.get_tensor('W' + str(i + 1))),
                                bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(i + 1)))))
                model.add(Activation('relu'))
            model.add(Dense(fc_node_number[weight_size - 1],
                            kernel_initializer=keras.initializers.Constant(reader.get_tensor('W' + str(weight_size))),
                            bias_initializer=keras.initializers.Constant(reader.get_tensor('b' + str(weight_size)))))

        self.model = model

    def predict(self, data):
        return self.model(data)

