# cs-280-speaker-recognizer
An end-to-end Speaker Recognition System written in Python using Mel Frequency Cepstral Coefficients (MFCC) and Spectral Subband Spectroid (SSC) for features and Support Vector Machine for classifier. A Mini Project requirement for CS 280 Intelligent Systems course at University of the Philippines Diliman AY 2015-2016 under Sir Prospero Naval. 

## Requirements
* Python 2.7.10
* Microphone capable of recording CD Quality (Stereo channels, 16 bit, 44100 Hz)

The following Python libraries not yet included in Python 2.7.10 Standard distribution needs to be installed:
* pyaudio
* tkintertable
* wave
* scipy
* libsvm
* [Python Speech Features](https://github.com/jameslyons/python_speech_features) by James Lyons

## Instructions
After installing required Python libraries
1. Clone and download the repository.
2. Go to the downloaded directory and run the application

  ```  
  python speaker-recognizer.py
  ```

