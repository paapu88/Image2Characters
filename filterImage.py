"""
Routines to make image more suitable for area recognition

self.filtered has the image after subsequent operations

to test:  python3 filterImage.py file.jpg
"""

import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np


class FilterImage():
    def __init__(self, npImage=None):
        """
        image can be read in as numpy array
        """
        self.img = npImage  # image as numpy array
        self.mser = cv2.MSER_create(_max_variation=10)
        self.regions = None
        self.otsu = None
        self.filtered = npImage
        if npImage is not None:
          self.imageY = self.img.shape[0]
          self.imageX = self.img.shape[1]


    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debugging: image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)
        self.imageY = self.img.shape[0]
        self.imageX = self.img.shape[1]
        self.filtered = self.img.copy()


    def getClone(self):
        return self.img.copy()

    def getFiltered(self):
        return self.filtered.copy()

    def reduce_colors(self, img, n):
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 100.0)
        K = n
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2


    def cleanImage(self):
        """ various trials to clean the image"""

        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        clone = self.filtered.copy()
        # clone = cv2.GaussianBlur(clone,(3,3),0)
        # d          Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
        # sigmaColor Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood
        #            will be mixed together, resulting in larger areas of semi-equal color.
        # sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as
        #            their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace.
        #            Otherwise, d is proportional to sigmaSpace.
        # cv2.imwrite('0-orig.png', clone)
        # blur = cv2.bilateralFilter(clone,d=5,sigmaColor=25, sigmaSpace=1)
        # cv2.imwrite('1-blur.png', blur)
        # equalized = cv2.equalizeHist(blur)
        # cv2.imwrite('2-equalized.png', equalized)
        th3 = cv2.adaptiveThreshold(clone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        reduced = cv2.cvtColor(self.reduce_colors(cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR), 2), cv2.COLOR_BGR2GRAY)
        cv2.imwrite('3-reduced.png', reduced)

        # ret, mask = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('4-mask.png', mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # mask2 = cv2.erode(reduced, kernel, iterations = 1)
        # cv2.imwrite('5-mask2.png', mask2)

        self.filtered = reduced

    def filterAdptiveThreshold(self):
        """http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html"""

        img = cv2.medianBlur(self.filtered.copy(),1)
        #ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        self.filtered = th3


    def filterOtsuManual(self):
        """ manually thresholding
        http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        """
        blur = cv2.GaussianBlur(self.filtered.copy(),(3,3),0)
        # find normalized_histogram, and its cumulative distribution function
        hist = cv2.calcHist([blur],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.max()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        thresh = -1
        for i in np.arange(1,256):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
            b1,b2 = np.hsplit(bins,[i]) # weights
            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
        # find otsu's threshold value with OpenCV function
        ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.filtered = otsu


    def filterOtsu(self, d=3, sigmaColor=3, dummy=50):
        """ http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html"""

        clone = self.filtered.copy()
        (thresh, im_bw) = cv2.threshold(clone, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.otsu = im_bw
        self.filtered = self.otsu
        return self.otsu

    def deBlur(self):
        """using laplace to get high? frequencies away"""
        self.filtered = cv2.fastNlMeansDenoising(self.filtered ,None,
                                                 h=20,templateWindowSize=3,searchWindowSize=3)

    def inPaint(self):
        """
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_inpainting/py_inpainting.html#inpainting
        """
        mask=np.ones(self.filtered.shape,dtype=np.ubyte)
        self.filtered = cv2.inpaint(self.filtered,mask,3,cv2.INPAINT_TELEA)

    def sharpen1(self):
        blurred = cv2.GaussianBlur(self.filtered, (3, 3), 1)
        self.filtered = cv2.addWeighted(self.filtered, 1.5, blurred, -0.5, 0)

    def sharpen2(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.filtered = cv2.filter2D(self.filtered, -1, kernel)

    def erosion(self):
        """
        http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        erosion and dilation are vice versa if the original letter is black
        """

        kernel = np.ones((3, 3), np.uint8)
        self.filtered = cv2.dilate(self.filtered,kernel,iterations = 1)

    def histogram(self):
        """calculate histogram based on sum over x-values of the image"""
        y=np.sum(self.getClone(),axis=1)
        plt.hist(y, bins='auto')  # plt.hist passes it's arguments to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    def showOriginalAndFiltered(self):
        """ show original and filtered image"""
        clone = self.getClone()
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(clone, cmap = 'gray', interpolation = 'bicubic')
        a.set_title('Before')
        a=fig.add_subplot(1,2,2)
        imgplot = plt.imshow(self.filtered, cmap = 'gray', interpolation = 'bicubic')
        a.set_title('After')
        plt.show()
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    def writeFiltered(self):
        cv2.imwrite('filtered'+sys.argv[1]+'.tif', self.filtered)


if __name__ == '__main__':
    import sys
    app = FilterImage()
    app.setImageFromFile(imageFileName=sys.argv[1])
    #app.histogram()
    app.erosion()
    #app.sharpen2()
    #app.inPaint()
    #app.deBlur()
    app.filterAdptiveThreshold()
    #app.filterOtsu()
    #app.filterOtsuManual()
    #app.cleanImage()
    app.showOriginalAndFiltered()
    app.writeFiltered()
