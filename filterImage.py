"""
Routines to make image more suitable for area recognition
self.filtered has the image after subsequent operations
to test:  python3 filterImage.py file.jpg
"""

import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal

class FilterImage():
    """
    Routines to make image more suitable for area recognition
    self.filtered has the image after subsequent operations
    to test:  python3 filterImage.py file.jpg
    """
    
    def __init__(self, npImage=None):
        """
        image can be read in as numpy array
        """
        self.img = npImage  # image as numpy array
        self.mser = cv2.MSER_create(_max_variation=10)
        self.regions = None
        self.otsu = None
        self.original = None
        self.filtered = npImage
        if npImage is not None:
          self.imageY = self.img.shape[0]
          self.imageX = self.img.shape[1]


    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debugging: image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        try:
            self.img = cv2.cvtColor(self.img, colorConversion)
        except:
            print("warning: no color conversion!")
        self.imageY = self.img.shape[0]
        self.imageX = self.img.shape[1]
        self.filtered = self.img.copy()
        self.original = self.img.copy()

    def setNumpyImage(self, image):
        """
        set image from numpy array
        """
        self.img = image

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

    def rotate(self, minAng=-10, maxAng=10):
        """ rotate to get max intensity along x direction (to get the two white stripes) """
        clone = self.getClone()
        rows,cols = clone.shape
        weights = []; angles = []
        #rotate image
        for angle in np.linspace(minAng, maxAng, 10*int(round(abs(maxAng)+abs(minAng)+1))):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            dst = cv2.warpAffine(clone,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            angles.append(angle)
            y=np.sum(dst,axis=1)
            #plt.plot(y)  # plt.hist passes it's arguments to np.histogram
            #plt.title("Histogram with 'auto' bins")
            #plt.show()
            weights.append(np.max(y))
        f = interpolate.interp1d(angles, weights)
        angles_tight = np.linspace(minAng, maxAng, 1+10*int(round(abs(maxAng)+abs(minAng))))
        faas = f(angles_tight)
        #plt.plot(angles_tight, faas)  # plt.hist passes it's arguments to np.histogram
        #plt.title("max white stripes")
        #plt.show()

        angles_tight = np.linspace(minAng, maxAng, 1+10*int(round(abs(maxAng)+abs(minAng))))
        faas = f(angles_tight)
        angle=angles_tight[np.argmax(faas)]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(clone,M,(cols,rows))

        #fig = plt.figure()
        #a=fig.add_subplot(1,2,1)
        #imgplot = plt.imshow(self.getClone(), cmap = 'gray', interpolation = 'bicubic')
        #a.set_title('before')
        #a=fig.add_subplot(1,2,2)
        #imgplot = plt.imshow(dst, cmap = 'gray', interpolation = 'bicubic')
        #a.set_title('After')
        #plt.show()
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # set the rotated image to be the current image
        self.setNumpyImage(dst)

    def cut_plate_peaks_inY(self):
        """get the two peaks in Y correspoinding the two white stripes in plate, cut the plate along the two stripes"""
        clone = self.filtered.copy()
        # get the peaks in intensity
        y=np.sum(clone,axis=1)
        #y=self.descew(y)
        peaks = signal.find_peaks_cwt(y, np.arange(10,20))
        #print(peaks)
        mymax = []
        for peak in peaks:
            mymax.append(y[peak])
        # get biggest value first
        mymax, peaks = zip(*sorted(zip(mymax, peaks)))
        #print(mymax)
        plt.plot(y)
        plt.title("vertical intensities")
        plt.show()
        # cut in y direction
        if peaks[-2]<peaks[-1]:
            cutted = clone[peaks[-2]:peaks[-1],:]
        else:
            cutted = clone[peaks[-1]:peaks[-2],:]
        #plt.imshow(cutted, cmap = 'gray', interpolation = 'bicubic')
        #plt.show()
        self.filtered = cutted


    def cut_plate_peaks_inX(self, threshold=0.8):
        """get the two peaks in X correspoinding the two white stripes in plate, cut the plate along the two stripes"""
        clone = self.filtered.copy()
        # get the peaks in intensity
        x=np.sum(clone,axis=0)
        #x = self.descew(x)
        peak_indexes = signal.find_peaks_cwt(x, np.arange(10,20))
        #print(peaks)
        peaks = []
        for peak_index in peak_indexes:
            peaks.append(x[peak_index])
        # get biggest value first
        peaks, peak_indexes = zip(*sorted(zip(peaks, peak_indexes)))
        print(peaks, peak_indexes)
        plt.plot(x)  # plt.hist passes it's arguments to np.histogram
        plt.title("vertical intensities")
        plt.show()
        avg=(peaks[-2]+peaks[-1])/2
        # get values that are > average of the brighest peaks times threshols (default 0.8)
        ok_peaks = []; ok_peak_indexes=[]
        for i, peak in zip(peak_indexes, peaks):
            if peak > threshold*avg:
                ok_peak_indexes.append(i)

        ok_peak_indexes = np.sort(ok_peak_indexes)
        #plt.plot(y)
        #plt.plot(peaks)
        #plt.show()
        # cut in y direction
        cutted = clone[:, ok_peak_indexes[0]:ok_peak_indexes[-1]]
        plt.imshow(cutted, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
        self.filtered = cutted

    def histogram(self):
        """calculate histogram based on sum over x-values of the image"""
        clone = self.getClone()
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(clone, cmap = 'gray', interpolation = 'bicubic')
        a.set_title('Before')
        a=fig.add_subplot(1,2,2)
        y=np.sum(self.getClone(),axis=1)
        plt.plot(y)  # plt.hist passes it's arguments to np.histogram
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

    def descew(self, x, method='log'):
        """try descewing data"""
        if method == 'sqrt':
            y = np.sqrt(x)
        elif method == 'log':
            y = np.log(x)
        else:
            raise NotImplementedError("skew method not implemented")
        return y

if __name__ == '__main__':
    import sys, glob
    from filterCharacterRegions import FilterCharacterRegions
    for imageFileName in glob.glob(sys.argv[1]):
        print("file:", imageFileName)
        app = FilterImage()
        app.setImageFromFile(imageFileName=imageFileName)

        #app.filterOtsu()
        app.rotate()
        app.cut_plate_peaks_inY()
        app.cut_plate_peaks_inX()
        app.showOriginalAndFiltered()

    sys.exit()
    app2 = FilterCharacterRegions()
    app.setNumpyImage(app2.descew_histo(area=app.getClone()))
    app.histogram()
    #app.erosion()
    #app.sharpen2()
    #app.inPaint()
    #app.deBlur()
    #app.filterAdptiveThreshold()
    #app.filterOtsuManual()
    #app.cleanImage()
    #app.showOriginalAndFiltered()
    #app.writeFiltered()
