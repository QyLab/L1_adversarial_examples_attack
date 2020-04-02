import tensorflow as tf
import numpy as np
import time
import random
import os

from setup_mnist import MNIST, MNISTModel
from setup_cifar_10 import CIFAR_10, CIFAR_10_Model
from setup_cifar_20 import CIFAR_20, CIFAR_20_Model
from setup_cifar_100 import CIFAR_100, CIFAR_100_Model

from l1_attack import cnn_EADL1
from l1_attack_cov import cnn_EADL1_cov
from en_attack import cnn_EADEN
from en_attack_cov import cnn_EADEN_cov
from only_l1 import only_l1


def generate_data(data, samples, targeted=True, start=0, inception=False):

    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def test_attack_main(model_path, norm, image_size, data_index, output_dir, cnn, channels, data_set,
                     en, beta, cov, only):
    with tf.Session() as sess:
        target = False

        if data_set == 0:
            batch_size = 1
            data, model = MNIST(), MNISTModel(model_path, image_size, cnn, channels)
        elif data_set == 1:
            batch_size = 9
            data, model = CIFAR_10(), CIFAR_10_Model(model_path, image_size, cnn, channels)
        elif data_set == 2:
            batch_size = 19
            data, model = CIFAR_20(), CIFAR_20_Model(model_path, image_size, cnn, channels)
        else:
            batch_size = 99
            data, model = CIFAR_100(), CIFAR_100_Model(model_path, image_size, cnn, channels)

        if cnn:
            if norm == 2:
                attack = cnn_CarliniL2(sess, model, batch_size=batch_size, max_iterations=1000, confidence=0)
            elif norm == 1:
                if only:
                    attack = only_l1(sess, model, batch_size=batch_size, max_iterations=1000, confidence=0)
                else:
                    if cov:
                        if en:
                            attack = cnn_EADEN_cov(sess, model, batch_size=batch_size, max_iterations=1000, beta=beta,
                                               abort_early=False, targeted=target)
                        else:
                            attack = cnn_EADL1_cov(sess, model, batch_size=batch_size, max_iterations=1000, beta=beta,
                                           abort_early=False, targeted=target)
                    else:
                        if en:
                            attack = cnn_EADEN(sess, model, batch_size=batch_size, max_iterations=1000, beta=beta,
                                               abort_early=False, targeted=target)
                        else:
                            attack = cnn_EADL1(sess, model, batch_size=batch_size, max_iterations=1000, beta=beta,
                                           abort_early=False, targeted=target)
            else:
                attack = cnn_CarliniLi(sess, model)
        else:
            if norm == 2:
                attack = dnn_CarliniL2(sess, model, batch_size=batch_size, max_iterations=1000, confidence=0,
                                       targeted=target)
            elif norm == 1:
                attack = dnn_EADL1(sess, model, batch_size=batch_size, max_iterations=1000, beta=1e-2,
                                   abort_early=False)
            else:
                attack = dnn_CarliniLi(sess, model)
            pass

        inputs, targets = generate_data(data, samples=1, targeted=target, start=data_index, inception=False)
        if cnn == 0:
            inputs = inputs.reshape([-1, image_size*image_size*channels])

        sess.run(tf.global_variables_initializer())
        ys = sess.run(model.predict(tf.cast(inputs, tf.float64)))
        print(ys[0])
        y_true = np.argmax(ys[0])

        time_start = time.time()
        adv = attack.attack(inputs, targets)
        time_end = time.time()

        print("Took", time_end-time_start, "seconds to run", len(inputs), "samples.")

        fid_image_adv = open(output_dir + '/image_adv_' + str(data_index) + '.txt', 'w')
        fid_image_adv_y = open(output_dir + '/image_adv_y_' + str(data_index) + '.txt', 'w')
        fid_result = open(output_dir + '/result_' + str(data_index) + '.txt', 'w')

        if not os.path.exists("adv_t"):  # 存放数据，便于matlab读取
            os.mkdir("adv_t")
        fid_image_adv_t = open('adv_t/image_adv_' + str(data_index) + '.txt', 'w')
        fid_image_adv_y_t = open('adv_t/image_adv_y_' + str(data_index) + '.txt', 'w')
        fid_result_t = open('adv_t/result_' + str(data_index) + '.txt', 'w')
        for i in range(len(adv)):
            if cnn:
                for j in range(0, image_size):
                    for k in range(0, image_size):
                        for l in range(0, channels):
                            fid_image_adv.write(str(adv[i][j][k][l]) + "\t")
                            fid_image_adv_t.write(str(adv[i][j][k][l]) + "\t")
            else:
                for j in range(0, image_size*image_size*channels):
                    fid_image_adv.write(str(adv[i][j]) + "\t")
                    fid_image_adv_t.write(str(adv[i][j]) + "\t")
            fid_image_adv.write("\n")
            fid_image_adv_t.write("\n")

            p_y = model.model.predict(adv[i:i + 1])[0]
            for j in range(0, len(p_y)):
                fid_image_adv_y.write(str(p_y[j])+"\t")
                fid_image_adv_y_t.write(str(p_y[j]) + "\t")
            fid_image_adv_y.write("\n")
            fid_image_adv_y_t.write("\n")
            if i < y_true:
                adv_y_t = i
            else:
                adv_y_t = i+1
            print(adv_y_t)
            if norm == 1:
                print("Total distortion:", np.sum(abs(adv[i] - inputs[i])))
                fid_result.write(str(adv_y_t) + '\t' + str(np.sum(abs(adv[i] - inputs[i]))) + "\t")
                fid_result_t.write(str(adv_y_t) + '\t' + str(np.sum(abs(adv[i] - inputs[i]))) + "\t")
            elif norm == 2:
                print("Total distortion:", np.sum((adv[i] - inputs[i]) ** 2) ** .5)
                fid_result.write(str(adv_y_t) + '\t' + str(np.linalg.norm((adv[i] - inputs[i]))) + "\t")
                fid_result_t.write(str(adv_y_t) + '\t' + str(np.linalg.norm((adv[i] - inputs[i]))) + "\t")
            else:
                print("Total distortion:", np.max(abs(adv[i] - inputs[i])))
                fid_result.write(str(adv_y_t) + '\t' + str(np.max(abs(adv[i] - inputs[i]))) + "\t")
                fid_result_t.write(str(adv_y_t) + '\t' + str(np.max(abs(adv[i] - inputs[i]))) + "\t")
            if target:
                if adv_y_t == np.argmax(p_y):
                    fid_result.write(str(1) + "\n")
                    fid_result_t.write(str(1) + "\n")
                else:
                    fid_result.write("0000000" + "\n")
                    fid_result_t.write("0000000" + "\n")
            else:
                if y_true != np.argmax(p_y):
                    fid_result.write(str(1) + "\n")
                    fid_result_t.write(str(1) + "\n")
                else:
                    fid_result.write("0000000" + "\n")
                    fid_result_t.write("0000000" + "\n")

        fid_image_adv.close()
        fid_image_adv_y.close()
        fid_result.close()
        fid_image_adv_t.close()
        fid_image_adv_y_t.close()
        fid_result_t.close()
