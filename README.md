# Image2Letters
from an image by haar cascade get plate,
then from plate get regions (by MSER) ,
from regions get letters/digits by SVM

Usage:
     image2characters.py vkz-825.jpg
     

# Haar cascade description:
https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

# The SVM files are copied as follows
cp TrainSVM/Letters/SvmDir/allSVM.txt.dict letters.dict
cp TrainSVM/Letters/SvmDir/digits_svm.dat letters_svm.dat
cp TrainSVM/Digits/SvmDir/allSVM.txt.dict digits.dict
cp TrainSVM/Digits/SvmDir/digits_svm.dat .


* opencv installation
examples are in
Kauppi:~/Programs/opencv-3.2.0/build/bin

see    http://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.html
opencv installation (because cv2.text needs opencv>= 3.0)
I downloaded 06.06.2017
https://github.com/opencv/opencv/archive/3.2.0.zip
https://github.com/opencv/opencv_contrib/archive/3.2.0.zip

sudo apt-get install apt-file
apt-file update
apt-file search gstreamer-base-1.0
sudo apt-get install libgstreamer1.0-dev
sudo apt-get install libgstreamer-plugins-base1.0-dev
sudo apt-get install libavresample-dev libavresample-ffmpeg2 libgphoto2-dev
sudo apt-get install libgoogle-glog-dev
sudo apt-get install libopenblas-dev liblapacke-dev checkinstall

Kauppi:~/Programs/opencv-3.2.0
cd ~/Programs/opencv-3.2.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/Programs/opencv_contrib-3.2.0/modules \
-D INSTALL_PYTHON_EXAMPLES=ON  -D ENABLE_PRECOMPILED_HEADERS=OFF \
-D BUILD_EXAMPLES=ON ..

make

sudo make install
sudo ldconfig

# for tesseract (not used for the moment)
# you need tesserocr (and tesseract, which hopefully comes automatically below))
apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
sudo -H pip3 install tesserocr


