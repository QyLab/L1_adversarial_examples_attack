import test_attack
import time
import os

all_data_set = [0]
all_data_index = [2, 3, 4, 5]
all_beta = [1]
all_en = [True]
all_cov = [True]
only_l1 = True
for i_cov in range(0, len(all_cov)):
    cov = all_cov[i_cov]
    for i_en in range(0, len(all_en)):
        en = all_en[i_en]
        for i_beta in range(0, len(all_beta)):
            beta = all_beta[i_beta]
            for i in range(0, len(all_data_set)):
                data_set = all_data_set[i]  # 0:mnist 1:cifar-10 2:cifar-20 3:cifar-100
                cnn = 1
                norm = 1
                for j in range(0, len(all_data_index)):
                    data_index = all_data_index[j]

                    if data_set == 0:
                        data_set_name = 'mnist'
                        image_size = 28
                        channels = 1
                    elif data_set == 1:
                        data_set_name = 'cifar_10'
                        image_size = 32
                        channels = 3
                    elif data_set == 2:
                        data_set_name = 'cifar_20'
                        image_size = 32
                        channels = 3
                    else:
                        data_set_name = 'cifar_100'
                        image_size = 32
                        channels = 3

                    if cnn:
                        nn_name = 'cnn'
                    else:
                        nn_name = 'dnn'

                    model_path = '../' + nn_name + '_' + data_set_name + '_LeNet/'

                    now = str(time.strftime('%Y%m%dT%H%M%S', time.localtime(time.time())))
                    # output_dir = "adv/" + now
                    if cov:
                        if en:
                            output_dir = "adv/" + str(norm) + 'norm/' + data_set_name + "_" + str(data_index) + "_" + "en" \
                                         + "_cov_" + str(beta)
                        else:
                            output_dir = "adv/" + str(norm) + 'norm/' + data_set_name + "_" + str(data_index) + "_" + "l1"\
                                         + "_cov_" + str(beta)
                    else:
                        if en:
                            output_dir = "adv/" + str(norm) + 'norm/' + data_set_name + "_" + str(data_index) + "_" + "en" \
                                         + "_" + str(beta)
                        else:
                            output_dir = "adv/" + str(norm) + 'norm/' + data_set_name + "_" + str(data_index) + "_" + "l1"\
                                         + "_" + str(beta)
                    if only_l1:
                        output_dir = output_dir + "_only"
                    if not os.path.exists("adv"):
                        os.mkdir("adv")
                    if not os.path.exists("adv/" + str(norm) + "norm"):
                        os.mkdir("adv/" + str(norm) + "norm")
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    test_attack.test_attack_main(model_path=model_path, norm=norm, image_size=image_size,
                                                 data_index=data_index,
                                                 output_dir=output_dir, cnn=cnn, channels=channels, data_set=data_set,
                                                 en=en,
                                                 beta=beta,
                                                 cov=cov,
                                                 only=only_l1)

                    fid_readme = open(output_dir + '/readme.txt', 'w')
                    fid_readme.write("cnn=" + str(cnn) + "\n")
                    fid_readme.write("norm=" + str(norm) + "\n")
                    fid_readme.write("data_index=" + str(data_index) + "\n")
                    fid_readme.write("image_size=" + str(image_size) + "\n")
                    fid_readme.write("channels=" + str(channels) + "\n")
                    fid_readme.write("data_set=" + str(data_set) + "\n")
                    fid_readme.close()
