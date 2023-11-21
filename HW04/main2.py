import numpy as np
import matplotlib.pyplot as plt
import math
import gzip
import struct
from tqdm import trange

def load_mnist(kind):
    """
    Load the Mnist dataset

    :return: labels, images, number of images, number of pixels
    :rtype: np.array, np.array, int, int
    :param kind: the kind of dataset (test or train)
    :type kind: string
    """
    labels_path = "./datasets/{kind}-labels-idx1-ubyte.gz".format(kind=kind)
    images_path = "./datasets/{kind}-images-idx3-ubyte.gz".format(kind=kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        _, _ = struct.unpack('>II', lbpath.read(8)) # big endian, 32-bit integer
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8) # read remaining datas as 8-bit integer list

    with gzip.open(images_path, 'rb') as imgpath:
        _, num_of_img, row, column = struct.unpack('>IIII', imgpath.read(16)) # big endian, 32-bit integer
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), row*column)  # read remaining datas as 8-bit integer [labels, row*col] list

    return labels, images, num_of_img, row*column

def init_prob(num_of_class, num_of_pxs):
    prob = np.zeros((num_of_class, num_of_pxs))
    for c in range(num_of_class):
        for px_idx in range(num_of_pxs):
            prob[c][px_idx] = 0.25 + np.random.uniform(0, 0.5)
    return prob

def init_prob_2(imgs, lbs, num_of_class, num_of_pxs):
    label_num = np.zeros(10)
    prob = np.zeros((num_of_class, num_of_pxs))

    for n in range(num_of_train):
        img = imgs[n]
        lb = int(lbs[n][0])
        label_num[lb] += 1
        prob[lb][np.where(img == 1)] += 1
    
    for c in range(num_of_class):
        prob[c] /= label_num[c] + 1e-5
        prob[c] = 0.25 + 0.5 * np.random.uniform(0, prob[c])

    return prob

def mul(arr_A, arr_B):
    return np.matmul(arr_A, arr_B)

def output_img(prob):
    for c in range(10):
        print("Class {:d}:".format(c))
        for i in range(num_of_pxs):
            x = 1 if prob[c][i] >= 0.5 else 0
            print("{:d} ".format(x), end="")
            if (i+1) % 28 == 0 and i > 1:
                print()
        print()

def test_print_img(img):
    for i in range(num_of_pxs):
        print("{:d} ".format(round(img[i])), end="")
        if (i+1) % 28 == 0 and i > 1:
            print()
    print()

def print_confusion_matrix(num_of_train, w, lbs, iter):
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
        print("\nSensitivity (Successfully predict number {:<d}): {:<f}".format(c, TP/(TP+FN)))
        print("Specificity (Successfully predict not number {:<d}): {:<f}\n".format(c, TN/(FP+TN)))
        print("\n------------------------------------------\n")
    print("Total iteration to converge: {:d}".format(iter))
    print("Total error rate: {:f}".format(err/num_of_train))

if __name__ == '__main__':
    # Load data
    train_lbs, train_imgs, num_of_train, num_of_pxs = load_mnist('train')
    num_of_train = 60000
    imgs = np.zeros((num_of_train, num_of_pxs))
    lbs = np.zeros((num_of_train, 1))
    iter = 0

    # Preprocess data
    for i in trange(num_of_train, desc="Preprocess Data"):
        img = train_imgs[i]
        lbs[i] = train_lbs[i]
        imgs[i][np.where(img > 127)] = 1 
        imgs[i][np.where(img <= 127)] = 0
        # test_print_img(imgs[i])

    # Initilize the probability in each pixel for 10 class (0~9) and lambda
    # prob = init_prob(10, num_of_pxs) 
    prob = init_prob_2(imgs, lbs, 10, num_of_pxs)
    output_img(prob)
    lam = np.full([10, 1], 0.1)
    alpha = 1e-8
    beta = 1e-8

    while True:
        # Initilize w
        w = np.zeros((num_of_train, 10))
        diff = 0

        # E step, generate w
        for n in trange(num_of_train, desc="No. of Iteration: {}; [E step]".format(iter)):
            for c in range(10):
                # P(Z_i=c, x_i=00001110... |theta), Z_i=0~9, x_i=0~1
                w[n][c] += np.log(lam[c])
                w[n][c] += np.sum((imgs[n]) * np.log(prob[c]))
                w[n][c] += np.sum((1-imgs[n]) * np.log(1-prob[c]))
            w[n] = np.exp(w[n])
            w[n] = w[n] / np.sum(w[n])
            
        # M step, update lambda, prob
        for c in trange(10, desc="No. of Iteration: {}; [M step]".format(iter)):
            lam[c] = (np.sum(w[:, c]) + alpha) / (num_of_train + alpha*10)
            wi = w[:, c].reshape(num_of_train, 1)
            _p = (mul(wi.T, imgs) + beta) / (np.sum(w[:, c]) + beta*28*28) # Calculate new p for each pixel j
            diff += np.sum(np.abs(prob[c] - _p[0]))  # Update diff
            prob[c] = _p[0]
    
        output_img(prob)
        print("No. of Iteration: {:d}, Difference: {:f}\n".format(iter, diff))
        print("------------------------------------------\n")

        iter += 1
        if iter > 15 or diff < 10: # If converge or times of iter > 2000, then break
            break

    print_confusion_matrix(num_of_train, w, lbs, iter)
