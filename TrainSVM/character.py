
'''
SVM and KNearest digit recognition.

Sample loads a dataset of characters as image and tehir labels as a txt file
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
	python3 ./character.py allSVM.tif allSVM.txt

    allSVM.tif  contains images of positive (and negative if binary) samples
    allSVM.txt  contains labels

'''


# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import numpy as np
from numpy.linalg import norm

# local modules
from common import clock, mosaic



SH = 18 # height of a character
SW = 12 # width of a character

# CLASS_N = 33  #number of characters (missing IOQÅÄÖ, digits one and zero are used instead of I and O in Fin plates)


def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def str2int(x):
    return ord(x)

def load_digits(fndata, fnlabels):
    import json
    print('loading "%s" ...' % fndata)
    digits_img = cv2.imread(fndata, 0)
    #n_one_char = int(round(digits_img.size[1]/SW))  # number of samples in one row, one row is for one char
    digits = split2d(digits_img, (SW, SH))
    with open(fnlabels, 'r') as f:
        lines = f.readlines()
    labels=[]
    for line in lines:
        labels.append(ord(line.split()[0]))

    labels = np.asarray(labels)
    # map ascii codes of characters to integers 0..34
    icount = 0
    labels_dict = {}
    for key in labels:
        if not(key in labels_dict.keys()):
            labels_dict[key]=icount
            icount = icount + 1
    # return labels as an mp array of numbers from 0 to 34
    labels = [labels_dict[l] for l in labels]
    labels = np.asarray(labels)

    # write dictionary to disc, so we get asciis of characters


    # load dict from file:
    #with open('/path/to/my_file.json', 'r') as f:
    #try:
    #    data = json.load(f)
    ## if the file is empty the ValueError will be thrown
    #except ValueError:
    #    data = {}

    # save to file:
    print(labels_dict)
    with open(fnlabels + '.dict', 'w') as f:
        for (key, value) in labels_dict.items():
            f.write(str(key)+' '+str(value)+'\n')
        #json.dump(labels_dict, f)

    #print(labels[0:222])
    #print(labels_dict)

    return digits, labels, labels_dict

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*0.5*(SH+SW)*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SW, SH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    CLASS_N = len(labels)
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('error: %.2f %%' % (err*100))

    confusion = np.zeros((CLASS_N, CLASS_N), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print('confusion matrix:')
    print(confusion)
    print()

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(digits):
    samples = []
    for img in digits:
        samples.append(np.reshape(digits, (-1, SH*SW))/255.0)
    return samples.astype(np.float32)

#def preprocess_simple(digits):
#    return np.float32(digits).reshape(-1, SH*SW) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    import sys
    print(__doc__)

    digits, labels, labels_dict = load_digits(sys.argv[1], sys.argv[2])  # filename for image containing data, filename for labels

    print('preprocessing...', digits.shape, labels.shape)
    # shuffle digits
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    #digits2 = list(map(deskew, digits))
    #samples = preprocess_hog(digits2)
    samples = preprocess_hog(digits)
    #samples = preprocess_simple(digits)


    train_n = int(0.9*len(samples))
    cv2.imshow('test set', mosaic(25, digits[train_n:]))
    #digits_train, digits_test = np.split(digits2, [train_n])
    digits_train, digits_test = np.split(digits, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('training KNearest...')
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('KNearest test', vis)

    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    print("SIZES, data, labels",samples_train.shape, labels_train.shape)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('SVM test', vis)
    print('saving SVM as "digits_svm.dat"...')
    model.save('digits_svm.dat')

    cv2.waitKey(0)
