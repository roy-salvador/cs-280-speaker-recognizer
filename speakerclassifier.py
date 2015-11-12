import pyaudio
import numpy
import scipy.io.wavfile as wav
from features import mfcc
import math

# Wave File Format
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
CHANNELS = 2                # Most computers have stereo channels instead of mono
CHUNK = 1024   
RECORDING_TEMPFILENAME = 'temp.wav'

# Frame Processing Settings
FRAME_LENGTH = 0.030 # sec
OVERLAP = 0.010 #sec
VOICE_ACTIVITY_THRESHOLD = 200  # This could be adjusted based on capability of microphone/audio system of the computer


def extract_mfcc(audioFile) :
  print 'Extracting MFCC Features of ' + audioFile
  mfccFeatures = []
  
  (rate,sig) = wav.read(audioFile)
    
  # Segmentation into frames
  #print sig
  print len(sig)
  frameLength = math.ceil(FRAME_LENGTH * rate)
  overlap = math.ceil(OVERLAP * rate)
  
  i = 0
  while (i + frameLength) <= len(sig) :
    currentFrame = sig[i:(i+frameLength)]
    
    # Apply Hamming Window based on number of channels
    # Stereo
    if len(currentFrame[0]) == 2 :
      hammingWindow = numpy.vstack(numpy.transpose(numpy.hamming(frameLength)))
    # Mono
    else :
      hammingWindow = numpy.transpose(numpy.hamming(frameLength))
    windowedFrame = numpy.multiply(currentFrame, hammingWindow)
    
    # Scrap frames not meeting threshold / frames with weak signals (indicative of no voice activity)
    if numpy.mean(numpy.absolute(windowedFrame)) > VOICE_ACTIVITY_THRESHOLD :
      # Get MFCC feature of the current frame
      mfccFeatures.append(mfcc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH))
    #print str(i) + ' ' + str(numpy.mean(numpy.absolute(windowedFrame)))
    
    
    i = i + overlap
  
  print len(mfccFeatures)
  