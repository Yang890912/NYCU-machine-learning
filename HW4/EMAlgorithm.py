import numpy as np
import matplotlib.pyplot as plt
import math
import gzip
import struct
from tqdm import trange

def load_mnist(kind):
    labels_path = "./datasets/{kind}-labels-idx1-ubyte.gz".format(kind=kind)
    images_path = "./datasets/{kind}-images-idx3-ubyte.gz".format(kind=kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        _, _ = struct.unpack('>II', lbpath.read(8)) # big endian, 32-bit integer
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8) # read remaining datas as 8-bit integer list

    with gzip.open(images_path, 'rb') as imgpath:
        _, num_of_img, row, column = struct.unpack('>IIII', imgpath.read(16)) # big endian, 32-bit integer
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), row * column)  # read remaining datas as 8-bit integer [labels, row*col] list

    return labels, images, num_of_img, row * column

def init_prob(num_of_class, num_of_pxs, imgs, lbs):
    prob = np.zeros((num_of_class, num_of_pxs))
    for c in range(num_of_class):
        for n in range(num_of_train):
            if int(lbs[n][0]) == c:
                img = imgs[n]
                prob[c][np.where(img == 1)] = 0.5 + np.random.uniform(0, 0.2)
                prob[c][np.where(img != 1)] = 0.2 + np.random.uniform(0, 0.2)
                break
    return prob

def mul(arr_A, arr_B):
    return np.matmul(arr_A, arr_B)

def print_img(num_of_pxs, prob):
    for c in range(10):
        print("Class {:d}:".format(c))
        for i in range(num_of_pxs):
            x = 1 if prob[c][i] >= 0.5 else 0
            print("{:d} ".format(x), end="")
            if (i + 1) % 28 == 0 and i > 1:
                print()
        print()

def print_labeled_img(num_of_train, imgs, lbs):
    print("\n-------------------------------------------------------\n")
    for c in range(10):
        print("Labeled class {:d}:".format(c))
        for n in range(num_of_train):
            if int(lbs[n][0]) == c:
                for i in range(num_of_pxs):
                    print("{:d} ".format(int(imgs[n][i])), end="")
                    if (i + 1) % 28 == 0 and i > 1:
                        print()
                print()
                break

def print_confusion_matrix(num_of_train, w, lbs, step):
    err = 0
    for c in range(10):
        TP, FN, FP, TN = 0, 0, 0, 0
        for n in range(num_of_train):
            pred = np.argmax(w[n])
            truth = int(lbs[n][0])
            TP += 1 if truth == c and pred == c else 0
            FN += 1 if truth == c and pred != c else 0
            FP += 1 if truth != c and pred == c else 0
            TN += 1 if truth != c and pred != c else 0
        err += FN
        print("\nConfusion Matrix {:<d}:\n".format(c))
        print("\t\t\t\tPredict number {:<d} \t Predict not number {:<d}".format(c, c))
        print("Is number {:<d}\t\t\t {:^5d}\t\t\t\t\t{:^5d}".format(c, TP, FN))
        print("Isn't number {:<d}\t\t {:^5d}\t\t\t\t\t{:^5d}".format(c, FP, TN))
        print("\nSensitivity (Successfully predict number {:<d}): {:<f}".format(c, TP / (TP + FN)))
        print("Specificity (Successfully predict not number {:<d}): {:<f}\n".format(c, TN / (FP + TN)))
        print("\n-------------------------------------------------------\n")
    print("Total iteration to converge: {:d}".format(step))
    print("Total error rate: {:f}".format(err / num_of_train))

if __name__ == '__main__':
    train_lbs, train_imgs, num_of_train, num_of_pxs = load_mnist('train') # load data
    num_of_train = 60000
    imgs = np.zeros((num_of_train, num_of_pxs))
    lbs = np.zeros((num_of_train, 1))

    # preprocess data
    for i in trange(num_of_train, desc="Preprocess Data"):
        img = train_imgs[i]
        lbs[i] = train_lbs[i]
        imgs[i][np.where(img > 127)] = 1    # binning the gray level value into two bins
        imgs[i][np.where(img <= 127)] = 0   #

    prob = init_prob(10, num_of_pxs, imgs, lbs) # initilize the probability for each pixel with 10 classes (0 ~ 9)
    lam = np.full([10, 1], 0.1) # initilize lambda
    alpha, beta = 1e-8, 1e-8    # avoid zero divide
    step = 0

    while True:
        diff = 0
        step += 1
        w = np.zeros((num_of_train, 10))    # initilize w

        # E step: generate responsibility w
        for n in trange(num_of_train, desc="No. of Iteration: {} [E step]".format(step)):
            for c in range(10):
                # calculate P(z_i = c, x_i = 00001110... | theta), z_i = 0 ~ 9, x_i = 0 ~ 1
                w[n][c] += np.log(lam[c].item())
                w[n][c] += np.sum((imgs[n]) * np.log(prob[c]))
                w[n][c] += np.sum((1 - imgs[n]) * np.log(1 - prob[c]))
            w[n] = np.exp(w[n])
            w[n] = w[n] / np.sum(w[n])
            
        # M step: update lambda, probability by MLE
        for c in trange(10, desc="No. of Iteration: {} [M step]".format(step)):
            lam[c] = (np.sum(w[:, c]) + alpha) / (num_of_train + alpha * 10)    # update lambda
            w_c = w[:, c].reshape(num_of_train, 1)
            p_new = (mul(w_c.T, imgs) + beta) / (np.sum(w[:, c]) + beta * 28 * 28)   # update probability for each pixel j
            diff += np.sum(np.abs(prob[c] - p_new[0]))  # update diff
            prob[c] = p_new[0]
    
        print_img(num_of_pxs, prob)
        print("No. of Iteration: {:d}, Difference: {:f}\n".format(step, diff))
        print("=======================================================\n")

        if step > 15 or diff < 10:  # if converge or times of step > 15, then break
            break    

    print_labeled_img(num_of_train, imgs, lbs)
    print_confusion_matrix(num_of_train, w, lbs, step)