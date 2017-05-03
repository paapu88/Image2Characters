.. image2character documentation master file, created by
   sphinx-quickstart on Wed May  3 10:55:15 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

Welcome to image2character's documentation!
===========================================

Code is also in `github <https://github.com/paapu88/Image2Characters.git>`_

Program Flow:
-------------

#. from an image by haar cascade get plate,

#. then from plate get regions (by MSER) ,

#. from regions get letters/digits by SVM/logistic regression

Usage:
------
::

   python3 image2characters.py filename


More Description:
-----------------

* The flow of the program is defined in image2characters.py.

* It uses modules:

   1. rekkariDetectionSave.py |br|
   - to find a region in the image containing the plate |br|
   (haar cascades of opencv)

   2. filterImage.py |br|
   - to make the image more clear

   3. filterCharacterRegions.py |br| (inherits from initialCharacterRegions.py) |br|
   - to get list of six-rectangles for possible plete-character regions of the image

   4. myClassifier.py |br|
   - to detect characters

.. toctree::
   :maxdepth: 2


Documentation for the Code
**************************
.. automodule:: ./rekkariDetectionSave.py
    :members: DetectPlate
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

