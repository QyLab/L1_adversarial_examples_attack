
import tensorflow as tf
import numpy as np
import os

import keras
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense,  Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNIST:
    def __init__(self):
        data_set_path = "../dataset/MNIST_data/"

        train_data = extract_data(data_set_path + "train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels(data_set_path + "train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data(data_set_path + "t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels(data_set_path + "t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    def __init__(self, restore, image_size=28, cnn=1, channels=1):
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
        self.num_labels = fc_node_number[len(fc_node_number)-1]
        model = Sequential()

        if cnn:
            kernel_size = np.loadtxt(self.data_dir + 'kernel_size.txt', dtype=int)
            kernel_number = np.loadtxt(self.data_dir + 'kernel_number.txt', dtype=int)
            pool_size = np.loadtxt(self.data_dir + 'pool_size.txt', dtype=int)
            padding = np.loadtxt(self.data_dir + 'padding.txt', dtype=int)
            features = np.loadtxt(self.data_dir + 'features.txt', dtype=int)

            # 输入层连接的
            model.add(Conv2D(kernel_number[0], (kernel_size[0], kernel_size[0]), input_shape=(self.image_size, self.image_size, self.num_channels),
                             kernel_initializer=keras.initializers.Constant(reader.get_tensor('W1')),
                             bias_initializer=keras.initializers.Constant(reader.get_tensor('b1'))))
            model.add(Activation('relu'))
            for i in range(1, weight_size-1):  # 最后一层没有relu
                if features[i] == 1:
                    model.add(Conv2D(kernel_number[i], (kernel_size[i], kernel_size[i]),
                                     kernel_initializer=keras.initializers.Constant(reader.get_tensor('W'+str(i+1))),
                                     bias_initializer=keras.initializers.Constant(reader.get_tensor('b'+str(i+1)))))
                    model.add(Activation('relu'))
                elif features[i] == 3:
                    model.add(MaxPooling2D(pool_size=(pool_size[i], pool_size[i])))
                elif features[i] == 2:
                    model.add(AveragePooling2D(pool_size=(pool_size[i], pool_size[i])))
                elif features[i] == 0:  #全连接
                    if features[i-1] != 0:  # 拉平
                        model.add(Flatten())
                    model.add(Dense(fc_node_number[i],
                                    kernel_initializer=keras.initializers.Constant(reader.get_tensor('W'+str(i+1))),
                                    bias_initializer=keras.initializers.Constant(reader.get_tensor('b'+str(i+1)))))
                    model.add(Activation('relu'))
            model.add(Dense(fc_node_number[weight_size-1],
                            kernel_initializer=keras.initializers.Constant(reader.get_tensor('W'+str(weight_size))),
                            bias_initializer=keras.initializers.Constant(reader.get_tensor('b'+str(weight_size)))))
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

