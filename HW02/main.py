import gzip
import struct
import math
import numpy as np


def load_mnist(kind):
    """
    Load the Mnist dataset

    :return: labels, images, number of images, number of pixels
    :rtype: np.array, np.array, int, int
    :param kind: the kind of dataset (test or train)
    :type kind: string
    """
    labels_path = './datasets/{kind}-labels-idx1-ubyte.gz'.format(kind=kind)
    images_path = './datasets/{kind}-images-idx3-ubyte.gz'.format(kind=kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        _, _ = struct.unpack('>II', lbpath.read(8)) # big endian, 32-bit integer
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8) # read remaining datas as 8-bit integer list

    with gzip.open(images_path, 'rb') as imgpath:
        _, num_of_img, row, column = struct.unpack('>IIII', imgpath.read(16)) # big endian, 32-bit integer
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), row*column)  # read remaining datas as 8-bit integer [labels, row*col] list

    return labels, images, num_of_img, row*column

def Discrete_add_peudocount(label_px_bin_freq):
    """
    Add the peudocount, trans 0 as 1
    """
    for lb in range(10):
        px_bin_freq = label_px_bin_freq[lb]
        for i in range(num_of_pixels):
            for j in range(32):
                if px_bin_freq[i][j] == 0:
                    px_bin_freq[i][j] = 1
    
def Discrete_print_img(label_px_bin_freq):
    for lb in range(10):
        px_bin_freq = label_px_bin_freq[lb]
        print('%d :' %(lb))
        for i, bin_freq in enumerate(px_bin_freq):
            print('0', end='') if np.argmax(bin_freq) < 16 else print('1', end='')
            if (i+1) % 28 == 0:
                print()
            
def Discrete_Naive_Bayes_classifier():
    """
    The Naive_Bayes_classifier (Discrete)

    Calculate the probability of such event under each label ex: (px1=bin4, px2=bin5, ... px784=bin6|label=1),
    
    """
    label_px_bin_freq = {lb: np.zeros((num_of_pixels, 32), np.uint32) for lb in range(10)} 
    label_num_ratio = np.zeros(10)

    # Establish label_px_bin_freq 
    for i in range(num_of_train):
        img = train_images[i]
        px_bin_freq = label_px_bin_freq[train_labels[i]]
        label_num_ratio[train_labels[i]] += 1
        for j, px in enumerate(img):
            px_bin_freq[j][math.floor(px/8)] += 1
        
    Discrete_add_peudocount(label_px_bin_freq)
    label_num_ratio /= num_of_train
    err = 0

    for i in range(num_of_test):
        img = test_images[i]
        post = np.zeros(10)
        for lb in range(10):
            post[lb] += math.log(label_num_ratio[lb])   # Convert probability multiplication to log addition
            px_bin_freq = label_px_bin_freq[lb]
            for j, px in enumerate(img):
                _sum = sum(px_bin_freq[j])
                _num = px_bin_freq[j][math.floor(px/8)]
                post[lb] += math.log(_num/_sum)         # Convert probability multiplication to log addition

        # The positive part of log of fraction is smaller, the fraction is larger
        post = post/sum(post)
        print('Postirior (in log scale):')
        for j in range(10):
            print('%d:  %.16f' %(j, post[j]))
        print('Prediction: %d, Ans: %d\n' %(np.argmin(post), test_labels[i]))
        if np.argmin(post) != test_labels[i]:
            err += 1

    Discrete_print_img(label_px_bin_freq)
    print('\nError rate: %f' %(err/num_of_test))

def Continuous_print_img(mean):
    for lb in range(10):
        print('%d :' %(lb))
        for i, j in enumerate(mean[lb]):
            print('0', end='') if j < 128 else print('1', end='')
            if (i+1) % 28 == 0:
                print()

def Continuous_Naive_Bayes_classifier():
    """
    The Naive_Bayes_classifier (Continuous)

    Calculate the probability of such event under each label ex: (px1=bin4, px2=bin5, ... px784=bin6|label=1),
    """
    mean = {lb: np.zeros(num_of_pixels) for lb in range(10)} 
    var = {lb: np.ones(num_of_pixels) for lb in range(10)} 
    label_num_ratio = np.zeros(10)

    # Calculate mean
    for i in range(num_of_train):
        img = train_images[i]
        label_num_ratio[train_labels[i]] += 1
        for j, px in enumerate(img):
            mean[train_labels[i]][j] += px
    
    for lb in range(10):
        mean[lb] /= label_num_ratio[lb]

    # Calculate variance
    for i in range(num_of_train):
        img = train_images[i]
        for j, px in enumerate(img):
            var[train_labels[i]][j] += (px - mean[train_labels[i]][j])**2

    for lb in range(10):
        var[lb] /= label_num_ratio[lb]

    label_num_ratio /= num_of_train
    err = 0

    for i in range(num_of_test):
        img = test_images[i]
        post = np.zeros(10)
        for lb in range(10):
            post[lb] += math.log(label_num_ratio[lb])   # Convert probability multiplication to log addition
            for j, px in enumerate(img):
                post[lb] += math.log(1 / (2 * math.pi * var[lb][j])**(1/2)) # Convert probability multiplication to log addition
                post[lb] += -1* (px-mean[lb][j])**(2) / (2*var[lb][j])      # Convert probability multiplication to log addition

        # The positive part of log of fraction is smaller, the fraction is larger
        post = post/sum(post)
        print('Postirior (in log scale):')
        for j in range(10):
            print('%d:  %.16f' %(j, post[j]))
        print('Prediction: %d, Ans: %d\n' %(np.argmin(post), test_labels[i]))
        if np.argmin(post) != test_labels[i]:
            err += 1
    
    Continuous_print_img(mean)
    print('\nError rate: %f' %(err/num_of_test))

if __name__ == '__main__':
    
    # Load data
    train_labels, train_images, num_of_train, num_of_pixels = load_mnist('train')
    test_labels, test_images, num_of_test, _ = load_mnist('t10k')

    num_of_train = 3000
    num_of_test = 3

    n = int(input("Please enter toggle option (0 or 1):\n"))

    if n == 0:
        Discrete_Naive_Bayes_classifier()
    else:
        Continuous_Naive_Bayes_classifier()
