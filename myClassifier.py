"""
predict a single character with SVM
inspired by
http://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html

At the moment (4/2017) letter and digit recognation works ok,
but binary classification NOT (whether we have a character in the box or not)

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

class Classifier():
    def __init__(self, npImage=None, NeuralNetworkFileName = None, logRegFileName=None, svmFileName=None, dictionaryFile=None,
                 sizeX=12, sizeY=18):
        """

        :param npImage: numpy array of image
        :param NeuralNetworkFileName: filename of trained neural network scikit recognizer
        :param logRegFileName: filename of trained scikit logistic regression recognizer
        :param svmFileName: filename of trained scikit support vector machine recognizer
        :param dictionaryFile: mapping from character recognizer to ascii
        :param sizeX: size of character in pixels in X
        :param sizeY: size of character in pixels in Y
        """
        self.asciiDict = {}
        if NeuralNetworkFileName is not None:
            self.setNeuralNetwork(MLPFileName=NeuralNetworkFileName)
        else:
            self.NeuralNetwork = None
        if logRegFileName is not None:
            self.setlogReg(logRegFileName=logRegFileName)
        else:
            self.logistic = None
        if svmFileName is not None:
            self.setSvmTrainedFile(svmFileName=svmFileName)
        if dictionaryFile is not None:
            self.setDictionary(dictionaryFile=dictionaryFile)
        self.img = npImage  # image as numpy array
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.plateString = None
        self.plateStrings = []                 # final string(s) of a plate(s)
        self.plateStringsProbabilities = None  # probability(ies) of the final string(s)
        self.char = None

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debugging:  image can be read also from file"""
        self.img = cv2.imread(imageFileName)
        try:
            self.img = cv2.cvtColor(self.img, colorConversion)
        except:
            print("warning: no color conversion!")

    def setNumpyImage(self, image):
        """
        set image from numpy array
        """
        self.img = image

    def setCharacter(self, rectangle=None):
        """ set the character to be recognized as numpy array"""
        if rectangle is None:
            self.char = cv2.resize(self.img.copy(),(self.sizeX, self.sizeY))
            print(self.char.shape)
        else:
            (x,y,w,h) = rectangle
            self.char = self.img.copy()[y:y+h,x:x+w]

    def getCharacter(self):
        """ get the current character """
        if self.char is not None:
            return self.char
        else:
            self.setCharacter()
            return self.char

    def showCharacter(self):
        """ debugging: show the current character"""
        print(self.char)
        plt.imshow(self.char, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def setNeuralNetwork(self, MLPFileName='neuralNetwork.pkl'):
        """read trained neural network"""
        self.NeuralNetwork=joblib.load(MLPFileName)

    def setlogReg(self, logRegFileName):
        """ load trained logistic regression classifier from a file """
        self.logistic = joblib.load(logRegFileName)

    def setSvmTrainedFile(self, svmFileName):
        """load trained svm classifier"""
        self.svm = cv2.ml.SVM_load(svmFileName)

    def setDictionary(self, dictionaryFile):
        """A dictionary containing mapping from labels of svm to ascii codes of letters or digits"""
        self.dictionaryFile = dictionaryFile
        with open(dictionaryFile, 'r') as f:
            lines=f.readlines()
        for line in lines:
            value, key = line.split()
            key = line.split()[1]
            value = int(line.split()[0])
            self.asciiDict[key] = value

    def deskew(self, img):
        """ descew from
        http://codingexodus.blogspot.fi/2013/06/moment-based-de-skewing.html
        """
        SZ=max(self.sizeX, self.sizeY)
        SZ2=int(round(SZ))
        resized = cv2.resize(img,(SZ, SZ))
        m = cv2.moments(resized)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        resized = cv2.warpAffine(resized,M,(SZ2, SZ2),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        rotatedImg=cv2.resize(resized,(self.sizeX, self.sizeY))

        plt.imshow(rotatedImg, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        return rotatedImg

    def preprocess_simple(self):
        """ no preprosesing for neural network """
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY)).astype(np.float32)
        self.sample = resized.reshape((1, -1))  # to 1d

    def preprocess_hog(self):
        """picking right features, for SVM and logistic regression """
        self.sample = None
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
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
        hist /= np.linalg.norm(hist) + eps
        self.sample = np.reshape(hist, (-1, len(hist))).astype(np.float32)

    def get_character_by_LogReg(self, binary=False):
        """ identify the character by logistic regression, give also probability """
        self.preprocess_hog()
        #self.preprocess_simple()
        label = self.logistic.predict(self.sample)
        #print(label[0])
        #print ("PROB:", self.logistic.predict_proba(self.sample)[0][label[0]])
        if binary:
            return label
        else:
            mychar = str(chr(self.asciiDict[str(label[0])]))
            return mychar, self.logistic.predict_proba(self.sample)[0][label[0]]

    def get_character_by_neural_network(self, binary=True):
        """ identify the character by neural network """
        #self.preprocess_hog()
        self.preprocess_simple()
        #print("SS", self.sample.shape)
        label = self.NeuralNetwork.predict(self.sample)
        if binary:
            return label
        else:
            mychar = str(chr(self.asciiDict[str(label[0])]))
            return mychar

    def get_character_by_SVM(self, binary=False):
        """ identify the character by support vector machine """
        self.preprocess_hog()
        ret, resp = self.svm.predict(self.sample)
        #print (ret, resp)
        #print("prob:", resp)
        label = int(round(resp.flatten()[0]))
        if binary:
            return label
        else:
            mychar = str(chr(self.asciiDict[str(label)]))
            return mychar

    def defineSixPlateCharactersbyLogReg(self, listOfListofRectangles,
                                 lettersLogRegFile='letters_logistic.pkl',
                                 lettersDictionaryFile='letters_logreg.dict',
                                 digitsLogRegFile='digits_logistic.pkl',
                                 digitsDictionaryFile='digits_logreg.dict'):
        """ By logistic regression: check all plates and in each plate go through every set of 6-rectangles
        give a result for each 6-rectange, for instance ABC-123 """
        from Image2Characters import __path__ as module_path
        self.plateStringsProbabilities = []
        # if there are more thatn one candidate for 6-chars, we predict them all...
        for plate in listOfListofRectangles:
            if len(plate) != 6:
                raise RuntimeError('only six character plates allowed in getSixPlateCharacters')
            string=''
            prob = 1.0
            # alphabets
            self.setlogReg(logRegFileName=module_path[0]+'/'+lettersLogRegFile)
            self.setDictionary(dictionaryFile=module_path[0]+'/'+lettersDictionaryFile)
            for rectangle in plate[0:3]:
                self.setCharacter(rectangle=rectangle)
                mychar, myprob = self.get_character_by_LogReg()
                string = string + mychar
                prob = prob * myprob
            # digits
            self.setlogReg(logRegFileName=module_path[0]+'/'+digitsLogRegFile)
            self.setDictionary(dictionaryFile=module_path[0]+'/'+digitsDictionaryFile)
            for rectangle in plate[3:6]:
                self.setCharacter(rectangle=rectangle)
                mychar, myprob = self.get_character_by_LogReg()
                string = string + mychar
                prob = prob * myprob
            self.plateString = (string[0:3]+'-'+string[3:6])
            #print(self.plateString)
            self.plateStrings.append(self.plateString)
            self.plateStringsProbabilities.append(prob)


    def defineSixPlateCharacters(self, listOfListofRectangles,
                                 lettersSvmFile='letters_svm.dat',
                                 lettersDictionaryFile='letters.dict',
                                 digitsSvmFile='digits_svm.dat',
                                 digitsDictionaryFile='digits.dict'):
        """ By support vector machine: check all plates and in each plate go through every set of 6-rectangles
        give a result for each 6-rectange, for instance ABC-123 """
        from Image2Characters import __path__ as module_path

        # if there are more thatn one candidate for 6-chars, we predict them all...
        for plate in listOfListofRectangles:
            if len(plate) != 6:
                raise RuntimeError('only six character plates allowed in getSixPlateCharacters')
            string=''
            # alphabets
            self.setSvmTrainedFile(svmFileName=module_path[0]+'/'+lettersSvmFile)
            self.setDictionary(dictionaryFile=module_path[0]+'/'+lettersDictionaryFile)
            for rectangle in plate[0:3]:
                self.setCharacter(rectangle=rectangle)
                string = string + self.get_character_by_SVM()
            # digits
            self.setSvmTrainedFile(svmFileName=module_path[0]+'/'+digitsSvmFile)
            self.setDictionary(dictionaryFile=module_path[0]+'/'+digitsDictionaryFile)
            for rectangle in plate[3:6]:
                self.setCharacter(rectangle=rectangle)
                string = string + self.get_character_by_SVM()
            self.plateString = (string[0:3]+'-'+string[3:6])
            #print(self.plateString)
            self.plateStrings.append(self.plateString)

    def getFinalStrings(self):
        """ give the final result """
        return self.plateStrings, self.plateStringsProbabilities

if __name__ == '__main__':
    import sys

    # all characters
    app = Classifier(NeuralNetworkFileName='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Characters/SvmDir/neuralNetwork.pkl')
    app.setDictionary(dictionaryFile='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Characters/SvmDir/allSVM.txt.dict')
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.setCharacter()
    print("result NN:",app.get_character_by_neural_network(binary=False))

    #all characters
    app2 = Classifier(logRegFileName='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Characters/SvmDir/logistic.pkl')
    app2.setDictionary(dictionaryFile='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Characters/SvmDir/allSVM.txt.dict')
    app2.setImageFromFile(imageFileName=sys.argv[1])
    app2.setCharacter()
    print("result LOGREG:",app2.get_character_by_LogReg(binary=False))

    #binary classifier
    app = Classifier(NeuralNetworkFileName='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Binary/SvmDir/neuralNetwork.pkl')
    app.setDictionary(dictionaryFile='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Binary/SvmDir/allSVM.txt.dict')
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.setCharacter()
    print("result NN:",app.get_character_by_neural_network(binary=True))

    #binary classifier
    app2 = Classifier(logRegFileName='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Binary/SvmDir/logistic.pkl')
    app2.setDictionary(dictionaryFile='/home/mka/PycharmProjects/Image2Characters/TrainSVM/Binary/SvmDir/allSVM.txt.dict')
    app2.setImageFromFile(imageFileName=sys.argv[1])
    app2.setCharacter()
    print("result LOGREG:",app2.get_character_by_LogReg(binary=True))