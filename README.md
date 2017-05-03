# https://github.com/paapu88/Image2Characters.git
from an image by haar cascade get plate,
then from plate get regions (by MSER) ,
from regions get letters/digits by SVM/öogistic regression

Usage:
     python3 image2characters.py filename


The flow of the program is defined in image2characters.py.
It uses modules
rekkariDetectionSave.py
- to find a region in the image containing the plate (Haar cascades of opencv)
filterImage.py
- to make the image more clear
filterCharacterRegions.py (inherits from initialCharacterRegions.py)
- to get list of six-rectangles for possible plete-character regions of the image
myClassifier.py
- to detect characters



Background:
# Haar cascade description:
https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
raw training data in mka@mka-HP:~/PycharmProjects/Rekkari/Training

# The SVM/logistic regression files are trained in
└── TrainSVM
    ├── Digits
    │   └── SvmDir
    ├── Letters
    │   └── SvmDir

copied as follows
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Letters/SvmDir/logistic.pkl letters_logistic.pkl
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Letters/SvmDir/allSVM.txt.dict letters_logreg.dict
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Digits//SvmDir/logistic.pkl digits_logistic.pkl
Kauppi:~/PycharmProjects/Image2Characters> cp TrainSVM/Digits//SvmDir/allSVM.txt.dict  digits_logreg.dict



