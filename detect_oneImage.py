# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

python3 ../detect_oneImage.py orig1-plateOnly-glb-172.jpg /home/mka/PycharmProjects/deep-anpr-x64-y32-finplate/weights3.npz

python3 ../detect_oneImage.py orig1-plateOnly-glb-172.jpg /home/mka/PycharmProjects/deep-anpr-x64-y32/weights.npz



"""

import collections
import itertools
import math
import sys

import cv2
import numpy as np
import tensorflow as tf

import Image2Characters.model

from scipy.optimize import minimize, fmin_l_bfgs_b, fmin_powell, \
    basinhopping, differential_evolution, brute

from matplotlib import pyplot as plt

__all__ = (
        'DIGITS',
        'LETTERS',
        'CHARS',
        'sigmoid',
        'softmax',
    )

import numpy


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS + '-'

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))
          

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.05):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew

class Detect():
    """ maximaze the probability of a plate as seen by the trained neural network
        we search the best window in the image
        (upperleft x and y, and width and height in the image)
        give the letters of the plate for this subimage """


    def __init__(self, image=None, param_vals=None):
        self.image = image            # image containing the number plate
        self.param_vals = param_vals  # neural network optimal values from train.py
        self.uly = 10                  # upper left y of the portion
        self.ulx = 10                 # upper left x of the portion
        self.lry = image.shape[0]-10   # width of the portion
        self.lrx = image.shape[1]-10  # heigth of the portion
        self.prob = None              # final probability
        self.best_letters = None           # final characters of the plage
        self.scale=1e-8               # trick to get scipy minimize work for integer function
        self.best = 1.0
        self.best_rectangle = None

    def setNpImage(self,image):
        self.image=image
        
    def get_best(self):
        return self.best, self.best_letters

    def get_best_rectangle(self):
        return self.best_rectangle

    def make_scaled_im(self, clone):
        return cv2.resize(clone, (model.WINDOW_SHAPE[1], model.WINDOW_SHAPE[0]))


    def maximise_prob(self):
        """ by scipy cg, find the portion of the image that gives max probability of a plate"""
        x0 = np.asarray([0.1,0.1,0.9,0.9])
        #direc=np.asarray([100,100,-100,-100])
        bnds = ((0.0, 1.0), (0.0, 1.0),
                (0.0, 1.0), (0.0, 1.0))

        rranges = (slice(0, self.image.shape[0], 10),
                   slice(0, self.image.shape[1], 10),
                   slice(0, self.image.shape[0], 10),
                   slice(0, self.image.shape[1], 10))
        print(bnds)
        print("x0",x0)
        res = brute(self.detect, rranges)
        # res = differential_evolution(func=self.detect, bounds=bnds, strategy='best1exp', popsize=5, mutation=1.99)
        #res = minimize(self.detect, x0, method='SLSQP', bounds=bnds,  options = {'eps': 0.5})
        #BASIN define the new step taking routine and pass it to basinhopping
        #BASIN take_step = RandomDisplacementBounds(0, 1)
        #BASIN minimizer_kwargs = dict(method="SLSQP", bounds=bnds, options={'maxiter':1})
        #BASIN res  = basinhopping(self.detect, x0, T=0.001, minimizer_kwargs=minimizer_kwargs,take_step=take_step)
        #res = minimize(self.detect, x0, method='SLSQP')
        #xx,yy,zz = fmin_powell(self.detect, x0, direc=direc)
        #xval,fval,other = fmin_l_bfgs_b(func=self.detect, x0=x0, approx_grad=True, bounds=bnds)
        #xval,fval,other = fmin_l_bfgs_b(func=g, x0=x0, approx_grad=True)

        print("result1", res)

    def letter_probs_to_code(self, letter_probs):
        return "".join(CHARS[i] for i in np.argmax(letter_probs, axis=1))

    def g2(self, x):
        res = x[0] + x[1] - x[2] - x[3]
        return res

    def get_prob_and_letters(self, image):
        """ for a single image, get a probability that a plate of roughly right size is there,
        get also characters of the plate """

        xx, yy, params = model.get_detect_model()

        #plt.imshow(scaled_im)
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()

        present_prob=0.0
        xx, yy, params = model.get_detect_model()
        # Execute the model at each scale.
        with tf.Session(config=tf.ConfigProto()) as sess:
            y_vals = []
            feed_dict = {xx: np.stack([image])}
            feed_dict.update(dict(zip(params, self.param_vals)))
            y_vals.append(sess.run(yy, feed_dict=feed_dict))

            # Interpret the results in terms of bounding boxes in the input image.
            # Do this by identifying windows (at all scales)
            # where the model predicts a
            # number plate has a greater than 50% probability of appearing.
            #
            # To obtain pixel coordinates,
            # the window coordinates are scaled according
            # to the stride size, and pixel coordinates.
            i=0; y_val = y_vals[0]
            for window_coords in np.argwhere(y_val[0, :, :, 0] > -math.log(1./0.001 - 1)):
                letter_probs = (y_val[0,
                            window_coords[0],
                                window_coords[1], 1:].reshape(
                                7, len(CHARS)))
                letter_probs = softmax(letter_probs)

                img_scale = float(1)

                present_prob = sigmoid(
                    y_val[0, window_coords[0], window_coords[1], 0])

                print(present_prob, self.letter_probs_to_code(letter_probs))
        if present_prob > 0.0:
            letters = self.letter_probs_to_code(letter_probs)
        else:
            letters = None

        return present_prob, letters

    def get_all_prob_letters(self, nstepBigger=16, nstepSmaller=6, stepPix=4):
        """ scale image and get corresponding probability of existence a proper plate and get also plate characters"""
        images = []
        probs = []
        letters = []
        # start from small
        width = self.image.shape[1]-nstepSmaller * stepPix
        height = self.image.shape[0]-nstepSmaller * stepPix
        ulx = width/2
        uly = height/2
        for istep in range(nstepBigger + nstepSmaller):
            # take slice
            scaled_im = self.image.copy()[uly:(uly+height), ulx:(ulx+width)]
            scaled_im = cv2.resize(scaled_im, (model.WINDOW_SHAPE[1], model.WINDOW_SHAPE[0]))
            present_prob, letters = self.get_prob_and_letters(scaled_im)
            probs.append(present_prob)
            letters.append(letters)
            images.append(image)

        plt.plot(probs)
        plt.title("Probs")
        plt.show()


    def detect(self, x):
        """
        Detect number plates in an image.
        :param x
            area in image xtopleft, ytopleft, width, height

        """

        xarea = x.copy()
        #xarea[0] = x[0]*self.image.shape[0]
        #xarea[2] = x[2]*self.image.shape[0]
        #xarea[1] = x[1]*self.image.shape[1]
        #xarea[3] = x[3]*self.image.shape[1]
        #xarea = xarea.astype(int)


        #small_image = self.image.copy()[xarea[1]:(xarea[1]+xarea[3]),
        #                    xarea[0]:(xarea[0] + xarea[2])]
        # Convert the image to various scales.
        #scaled_im = self.make_scaled_im(small_image.copy())
        scaled_im = self.make_scaled_im(self.image.copy())

        # Load the model which detects number plates over a sliding window.
        xx, yy, params = model.get_detect_model()

        plt.imshow(scaled_im)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

        present_prob=0.0

        # Execute the model at each scale.
        with tf.Session(config=tf.ConfigProto()) as sess:
            y_vals = []
            feed_dict = {xx: np.stack([scaled_im])}
            feed_dict.update(dict(zip(params, self.param_vals)))
            y_vals.append(sess.run(yy, feed_dict=feed_dict))

            # Interpret the results in terms of bounding boxes in the input image.
            # Do this by identifying windows (at all scales)
            # where the model predicts a
            # number plate has a greater than 50% probability of appearing.
            #
            # To obtain pixel coordinates,
            # the window coordinates are scaled according
            # to the stride size, and pixel coordinates.
            i=0; y_val = y_vals[0]
            for window_coords in np.argwhere(y_val[0, :, :, 0] > -math.log(1./0.001 - 1)):
                letter_probs = (y_val[0,
                            window_coords[0],
                                window_coords[1], 1:].reshape(
                                7, len(CHARS)))
                letter_probs = softmax(letter_probs)

                img_scale = float(1)

                present_prob = sigmoid(
                    y_val[0, window_coords[0], window_coords[1], 0])

                print(present_prob, self.letter_probs_to_code(letter_probs))
        if present_prob > 0.0:
            letters = self.letter_probs_to_code(letter_probs)
        else:
            letters = None
        result = 1.0-present_prob
        if result < self.best:
            self.best = 1.0-present_prob
            self.best_letters = letters
            self.best_rectangle = xarea
        print("returning",result)
        return result


if __name__ == "__main__":
    im_gray = cv2.imread(sys.argv[1])[:, :, 0].astype(numpy.float32) / 255.
    #im = cv2.imread(sys.argv[1])
    #try:
    #    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #except:
    #    im_gray = im 

    f = np.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    print(im_gray.shape)
    #plt.imshow(im_gray)
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    app = Detect(image=im_gray, param_vals = param_vals)
    #app.maximise_prob()
    #detect(im_gray[27:120,140:480], param_vals)
    app.detect(im_gray)


