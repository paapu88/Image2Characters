"""
From an image get a list of possible characters of a licence plate,
There may be more than one plate in the image,
we hopefully get all licence plates.

Typically one also gets extra false positive plates

getting string of characters of a licence plate

cd  /home/mka/PycharmProjects/Image2Letters/test

Usage:
    python3 image2characters.py "plate.yuv"
"""
import sys

from rekkariDetectionSave import DetectPlate
# from filterImage import FilterImage
from filterCharacterRegions import FilterCharacterRegions
from initialCharacterRegions import InitialCharacterRegions
# from myTesseract import MyTesseract
from myClassifier import Classifier
import glob
import cv2


class image2Characters():
    """ from an input file or yuv numpy array get array of strings representing
    characters in (a) number plate(s) """
    def __init__(self, npImage=None):
        self.img = npImage  # image as numpy array

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debuggin image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)

    def getChars(self):
        """
        From Image to list of strings, representing characters of (a) number plate(s)
        """
        myChars = []
        app1 = DetectPlate(npImage=self.img)
        plates = app1.getNpPlates()
        #app1.showPlates()
        #app1.writePlates(name='plateOnly-'+sys.argv[1])
        #print(file+' number of plates found '+ str(len(plates)))
        for plate in plates:
            # from a plate image to list of six-rectangles
            #app2 = FilterImage(npImage=plate)
            #plate = app2.filterOtsu()
            app3 = FilterCharacterRegions(npImage=plate)
            platesWithCharacterRegions = app3.imageToPlatesWithCharacterRegions()
            app5 = Classifier(npImage=plate)
            #app3.showImage()
            app5.defineSixPlateCharacters(platesWithCharacterRegions)
            myChars = myChars + app5.getFinalStrings()
        return myChars


if __name__ == '__main__':
    import sys, glob

    files=glob.glob(sys.argv[1])
    print(files)
    if len(files)==0:
        raise FileNotFoundError('no files with search term: '+sys.argv[1])
    app = image2Characters()
    for file in files:
        app.setImageFromFile(imageFileName=file)
        print("Image, plate(s): ",file, app.getChars())
