import gzip
import struct
import math
import numpy as np
from tqdm import tqdm

def load_data(kind):
    labels_path = "./datasets/{kind}-labels-idx1-ubyte.gz".format(kind=kind)
    images_path = "./datasets/{kind}-images-idx3-ubyte.gz".format(kind=kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        _, _ = struct.unpack('>II', lbpath.read(8))             # big endian, 32-bit integer
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)   # read remaining datas as 8-bit integer list

    with gzip.open(images_path, 'rb') as imgpath:
        _, num_of_img, row, column = struct.unpack('>IIII', imgpath.read(16))                   # big endian, 32-bit integer
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), row*column) # read remaining datas as 8-bit integer [labels, row*col] list

    return labels, images, num_of_img, row*column

def add_peudo_count(label_pixel_bin_counts):
    for lb in range(10):
        pixel_bin_counts = label_pixel_bin_counts[lb]
        for i in range(num_of_pixels):
            for j in range(32): # 32 bins
                if pixel_bin_counts[i][j] == 0:
                    pixel_bin_counts[i][j] += 1
    
def disc_print_img(label_pixel_bin_counts):
    for lb in range(10):
        pixel_bin_counts = label_pixel_bin_counts[lb]
        print('\n%d :' %(lb))
        for i, bin_counts in enumerate(pixel_bin_counts):
            print('0', end='') if np.argmax(bin_counts) < 16 else print('1', end='')
            if (i+1) % 28 == 0:
                print()
            
def disc_naive_bayes_classifier():
    """
    The Naive Bayes classifier (discrete):

        Calculate the probability of such event under each label 
        
        Ex: (px_1 = bin4, px_2 = bin5, ... px_784 = bin6 | label = 1)
    
    """
    label_pixel_bin_counts = {lb: np.zeros((num_of_pixels, 32), np.uint32) for lb in range(10)} 
    label_num_ratio = np.zeros(10)  # prior

    # establish label_pixel_bin_counts (train) 
    for i in tqdm(range(num_of_train)):
        img = train_images[i]
        pixel_bin_counts = label_pixel_bin_counts[train_labels[i]]
        label_num_ratio[train_labels[i]] += 1
        for j, px in enumerate(img):
            pixel_bin_counts[j][math.floor(px/8)] += 1
        
    add_peudo_count(label_pixel_bin_counts)
    label_num_ratio /= num_of_train
    err = 0

    # test
    for i in range(num_of_test):
        img = test_images[i]
        post = np.zeros(10)
        for lb in range(10):
            post[lb] += math.log(label_num_ratio[lb])   # log addition
            pixel_bin_counts = label_pixel_bin_counts[lb]
            for j, px in enumerate(img):
                _sum = sum(pixel_bin_counts[j])
                _num = pixel_bin_counts[j][math.floor(px/8)]
                post[lb] += math.log(_num / _sum)   # log addition

        post /= sum(post)
        print('Postirior (in log scale):')
        for j in range(10):
            print('%d:  %.16f' %(j, post[j]))
        print('Prediction: %d, Ans: %d\n' %(np.argmin(post), test_labels[i]))   # the positive part of log of fraction is smaller, the fraction is larger
        if np.argmin(post) != test_labels[i]:
            err += 1
    
    # output Naive Bayes classifier
    disc_print_img(label_pixel_bin_counts)
    print('\nError rate: %f' %(err / num_of_test))


### Continuous ###

def cont_print_img(mean):
    for lb in range(10):
        print('\n%d :' %(lb))
        for i, j in enumerate(mean[lb]):
            print('0', end='') if j < 128 else print('1', end='')
            if (i+1) % 28 == 0:
                print()

def cont_naive_bayes_classifier():
    """
    The Naive Bayes classifier (continuous):

        Calculate the probability of such event under each label 
        
        Ex: (px_1 = bin4, px_2 = bin5, ... px_784 = bin6 | label = 1)

    """
    mean = {lb: np.zeros(num_of_pixels) for lb in range(10)} 
    var = {lb: np.ones(num_of_pixels) for lb in range(10)} 
    label_num_ratio = np.zeros(10)  # prior

    # calculate mean (train)
    for i in tqdm(range(num_of_train)):
        img, lb = train_images[i], train_labels[i]
        label_num_ratio[lb] += 1
        for j, px in enumerate(img):
            mean[lb][j] += px
    
    for lb in range(10):
        mean[lb] /= label_num_ratio[lb]

    # calculate variance (train)
    for i in tqdm(range(num_of_train)):
        img, lb = train_images[i], train_labels[i]
        for j, px in enumerate(img):
            var[lb][j] += (px - mean[lb][j])**2

    for lb in range(10):
        var[lb] /= label_num_ratio[lb]

    label_num_ratio /= num_of_train
    err = 0

    # test
    for i in range(num_of_test):
        img = test_images[i]
        post = np.zeros(10)
        for lb in range(10):
            post[lb] += math.log(label_num_ratio[lb])   # log addition
            for j, px in enumerate(img):
                post[lb] += -0.5 * math.log(2 * math.pi * var[lb][j]) # log addition
                post[lb] += -1 * (px - mean[lb][j])**(2) / (2 * var[lb][j]) # log addition

        post /= sum(post)
        print('Postirior (in log scale):')
        for j in range(10):
            print('%d:  %.16f' %(j, post[j]))
        print('Prediction: %d, Ans: %d\n' %(np.argmin(post), test_labels[i])) # the positive part of log of fraction is smaller, then the fraction is larger
        if np.argmin(post) != test_labels[i]:
            err += 1
    
    # output Naive Bayes classifier
    cont_print_img(mean)
    print('\nError rate: %f' %(err / num_of_test))

if __name__ == '__main__':
    train_labels, train_images, num_of_train, num_of_pixels = load_data('train')   # load trainning data
    test_labels, test_images, num_of_test, _ = load_data('t10k')                   # load test data
    num_of_train, num_of_test = 5000, 40

    if int(input("Please enter toggle option (0 or 1):\n")) == 0:  
        disc_naive_bayes_classifier()
    else: 
        cont_naive_bayes_classifier()
