"""
From an image get a list of possible licence plate strings.
The first one has the highest probability.

Usage:
    python3 image2characters.py "plate.jpg"

    or from other python modules:

    from image2characters import image2Characters
    app = image2Characters(npImage=myNParr)
    app.getChars()

"""
import sys

from Image2Characters.rekkariDetectionSave import DetectPlate
from filterImage import FilterImage
from Image2Characters.filterCharacterRegions import FilterCharacterRegions
from Image2Characters.initialCharacterRegions import InitialCharacterRegions
# from myTesseract import MyTesseract
from Image2Characters.myClassifier import Classifier
import glob
import cv2


class image2Characters():
    """ 
    from an input file or yuv numpy array get an array of strings 
    representing characters in (a) number plate(s) 
    """
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
        from Image2Characters import __path__ as module_path
        
        myChars = []
        myProb = []
        app1 = DetectPlate(trainedHaarFileName=module_path[0]+'/rekkari.xml',
                           npImage=self.img)
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
            app5.defineSixPlateCharactersbyLogReg(platesWithCharacterRegions)
            plate_chars, plate_probability = app5.getFinalStrings()
            myChars = myChars + plate_chars
            myProb = myProb + plate_probability

        # sort so that most probable comes first
        if any(myProb) is None:
            return myChars
        else:
            myProb, myChars = zip(*sorted(zip(myProb, myChars)))
            return myChars[::-1]

    def getCharsByNeuralNetwork(self):
        """
        get the caharcters of a plate by neural network,
        the image is first carefully filtered, so it only contains the actual plate
        """
        from Image2Characters import __path__ as module_path
        app1 = DetectPlate(trainedHaarFileName=module_path[0]+'/rekkari.xml',
                           npImage=self.img)
        plates = app1.getNpPlates()  # get the actual numpy arrays
        app3 = FilterImage()
        app2 = FilterCharacterRegions()
        for plate in plates:
            #app2.setNumpyImage(image=plate)
            #platesWithCharacterRegions = app2.imageToPlatesWithCharacterRegions()
            app3.setNumpyImage(image=plate)
            app3.rotate()
            app3.cut_plate_peaks_inY()
            app3.cut_plate_peaks_inX()
            app3.showOriginalAndFiltered()



if __name__ == '__main__':
    import sys, glob

    files=glob.glob(sys.argv[1])
    # print(files)
    if len(files)==0:
        raise FileNotFoundError('no files with search term: '+sys.argv[1])
    app = image2Characters()
    for file in files:
        app.setImageFromFile(imageFileName=file)
        print("Image, plate(s): ",file, app.getChars())
        app.getCharsByNeuralNetwork()
