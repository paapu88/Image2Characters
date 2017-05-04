"""
A collection of methods that find rectangles that 
may contain characters in a plate

"""


import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np


class InitialCharacterRegions():
    """
    A collection of methods that find rectangles that 
    may contain characters in a plate
    """
    def __init__(self, npImage=None):
        self.img = npImage  # image as numpy array
        self.mser = cv2.MSER_create(_max_variation=10)
        self.regions = None
        if npImage is not None:
            self.imageY = self.img.shape[0]
            self.imageX = self.img.shape[1]

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debugging: image can be read from file also"""
        import os
        if not os.path.isfile(imageFileName):
            raise FileNotFoundError('NO imagefile with name: ' + imageFileName)
        self.img = cv2.imread(imageFileName)
        try:
            self.img = cv2.cvtColor(self.img, colorConversion)
        except:
            print("Warning: color conversion failed, "+str(colorConversion))
        self.imageY = self.img.shape[0]
        self.imageX = self.img.shape[1]

    def getClone(self):
        return self.img.copy()

    def getRegions(self):
        return self.regions

    def getInitialRegionsFast(self):
        """
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html#fast
        """
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        # find and draw the keypoints
        clone = self.img.copy()
        kp = fast.detect(clone, None)
        clone = cv2.drawKeypoints(clone, kp, clone)
        plt.imshow(clone, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        print(len(kp))

    def getInitialRegionsMser(self):
        """ get rectangles of possible characters
            http://docs.opencv.org/trunk/d3/d28/classcv_1_1MSER.html
        """
        # tuple is used instead of list because we need make sets of tuples
        self.regions = tuple(self.mser.detectRegions(self.img)[-1])

    def getInitialRegionsSIFT(self):
        """surface detection, works only with contributed stuff in opencv3
        try also SURF
        """
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = sift.detectAndCompute(self.img, None)
        self.img=cv2.drawKeypoints(self.img,kps,outImage=None)
        plt.imshow(self.img, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        #self.img = cv2.drawKeypoints(self.img, kps, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def getInitialRegionsByContours(self):
        clone = self.getClone()
        clone = cv2.bitwise_not(clone)
        ret, clone = cv2.threshold(clone, 30, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(clone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        # cv2.drawContours(self.img, contours, -1, (0,255,0), 1)
        self.regions = []
        for contour in contours:
            self.regions.append(cv2.boundingRect(contour))

    def showAllRectangles(self, clone=None, regions=None):
        """ show image with current rectangles on it"""
        if self.regions is None and regions is None:
            raise RuntimeError("No rectangles, did you remember to search them by some method in InitialCharacterRegions?")
        if clone is None:
            clone = self.getClone()
        if regions is None:
            regions=self.regions
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2RGB)
        for (x,y,w,h) in regions:
            cv2.rectangle(clone,(x,y),(x+w,y+h),(255,0, 0),3)
        plt.imshow(clone)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def writeAllRectangles(self, clone=None, regions=None):
        """ write all current rectangles to disk individually """
        if self.regions is None and regions is None:
            raise RuntimeError("No rectangles, did you remember to search them by some method in InitialCharacterRegions?")
        if clone is None:
            clone = self.getClone()
        if regions is None:
            regions=self.regions
        for i, (x,y,w,h) in enumerate(self.regions):
            roi_gray = clone[y:y+h, x:x+w]
            cv2.imwrite(str(i)+'-'+sys.argv[1]+'.tif', roi_gray)

if __name__ == '__main__':
    import sys
    app = InitialCharacterRegions()
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.getInitialRegionsMser()
    #app.getInitialRegionsByContours()
    #app.getInitialRegionsSIFT()
    app.showAllRectangles()
    app.writeAllRectangles()
