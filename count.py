import os


if not os.path.exists("count"):
    os.mkdir("count")
name1 = ["en", "en_cov", "l1", "l1_cov"]
name2 = ["1e-05", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000"]

for i in range(0, len(name1)):
    file1 = open("count/" + name1[i] + '.txt', 'w')
    for j in range(0, len(name2)):
        file2 = open("adv/1norm/mnist_0_" + name1[i] + "_" + name2[j] + "/result_0.txt", "r")
        file1.write(file2.readline())
        file2.close()
    file1.close()
