# cs-280-speaker-recognizer
An end-to-end Speaker Recognition System written in Python using Mel Frequency Cepstral Coefficients (MFCC) and Spectral Subband Spectroid (SSC) for features. A Mini Project requirement for CS 280 Intelligent Systems course at University of the Philippines Diliman AY 2015-2016 under Sir Prospero Naval. 

## Types of Classifiers Included
* Binary SVM per user
* One vs Rest Multi Class SVM
* Decision Tree
* Gaussian Naive Bayes Classifier

## Requirements
* Python 2.7.10
* Microphone/Audio Input System capable of recording CD Quality (Stereo channels, 16 bit, 44100 Hz)

The following Python libraries not yet included in Python 2.7.10 Standard distribution needs to be installed:
* pyaudio
* tkintertable
* wave
* scipy
* libsvm
* scikit-learn
* [Python Speech Features](https://github.com/jameslyons/python_speech_features) by James Lyons

## Instructions
After installing required Python libraries

1. Clone and download the repository.
2. Go to the downloaded directory and run the application

  ```  
  python speaker-recognizer.py
  ```

