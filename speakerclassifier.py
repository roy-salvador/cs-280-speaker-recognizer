import pyaudio
import numpy
import scipy.io.wavfile as wav
from features import mfcc
import math
from svmutil import *

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

# SVM Training Data
USERS = []
RAW_TRAINING_MFCC_FEATURES = []
RAW_TRAINING_LABELS = []

# Initialization
def init() :
  global USERS
  global RAW_TRAINING_MFCC_FEATURES
  global RAW_TRAINING_LABELS
  
  # Load Users
  print 'Initializing saved users'
  usersDB = open('model\users.csv', "r")
  for user in usersDB :
    USERS.append(user.strip())
  usersDB.close()
  
  # Load Raw Training Data
  print 'Initializing saved training data'
  trainFile = open("model/raw_mfcc_training.csv", "r")
  for line in trainFile:
    j=0
    mfccVector = []
    for value in line.strip().split(',') :
      if j==0 :
        RAW_TRAINING_LABELS.append(int(value))
      else :
        mfccVector.append(float(value)) 
      j = j+1
    RAW_TRAINING_MFCC_FEATURES.append(mfccVector)	
  trainFile.close()
    

# Extracts the MFCC Feature vectors of the audio file
def extract_mfcc(audioFile) :
  print 'Extracting MFCC Features of ' + audioFile
  mfccFeatures = []
  
  (rate,sig) = wav.read(audioFile)
    
  # Segmentation into frames
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
      featureVectors = mfcc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH)
      mfccFeatures = mfccFeatures + featureVectors.tolist() 
      
    #print str(i) + ' ' + str(numpy.mean(numpy.absolute(windowedFrame)))
    i = i + overlap
 
  print 'There are ' + str(len(mfccFeatures)) + ' extracted MFCC feature vectors.'
  return mfccFeatures

# Returns the label vector for the binary classifier of the user
#   1 tagged as MFCC feature of the user
#  -1 tagged as MFCC feature is not the user's
def getLabelVectorForUser(userID) :
  global RAW_TRAINING_LABELS
  labelVector = []
  for id in RAW_TRAINING_LABELS :
    if id == userID :
      labelVector.append(1)
    else :
      labelVector.append(-1)
  
  return labelVector
  
# Trains all the user SVMS with the new training data of the new user
def train_user_svms(name, audioFile) :
  global RAW_TRAINING_MFCC_FEATURES
  global RAW_TRAINING_LABELS
  global USERS

  # Add and save the new user
  print '============================================================================================='
  print 'Adding new user ' + name
  userID = len(USERS)
  USERS.append(name)
  # Persist user
  usersFile = open("model/users.csv", "w")
  for userName in USERS :
    usersFile.write(userName + '\n')
  usersFile.close()
  
  # Extract and save user MFCC Features
  mfccFeatures = extract_mfcc(audioFile)
  RAW_TRAINING_MFCC_FEATURES = RAW_TRAINING_MFCC_FEATURES + mfccFeatures
  i=0
  while i < len(mfccFeatures) :
    RAW_TRAINING_LABELS.append(userID)
    i=i+1
  # Persist training data
  print 'Saving obtained training data'
  mfccFile = open("model/raw_mfcc_training.csv", "w")
  i=0
  while i<len(RAW_TRAINING_LABELS) :
    mfccFile.write(str(RAW_TRAINING_LABELS[i]) + ',' + str(RAW_TRAINING_MFCC_FEATURES[i]).lstrip('[').rstrip(']') + '\n')
    i = i+1
  mfccFile.close()
  
  # Train an SVM for each user
  i=0
  while i < len(USERS) :
    print '************************************************************'
    print 'Training SVM #' + str(i) + ' for user ' + USERS[i]
    prob = svm_problem(getLabelVectorForUser(i), RAW_TRAINING_MFCC_FEATURES)
    param = svm_parameter('-t 2')  #use RBF kernel
    m = svm_train(prob, param)
    svm_save_model('model/user' + str(i) + '.model', m)
    i = i+1
    
  print '************************************************************'
  print 'Training Complete'
 
# Returns the SVM prediction accuracies in the dict format the tktable can use 
def classify_audio(audioFile) : 
  results = {}
  print '============================================================================================='
  print 'Predicting which user is speaking in the audio file'
  
  # Extract MFCC Features and initialize label vector
  mfccFeatures = extract_mfcc(audioFile)
  y = []
  for mf in mfccFeatures:
    y.append(1)   # assume mfcc vector belongs to the user
  
  # Get accuracy for every user
  i=0
  while i < len(USERS) :
    print '************************************************************'
    print 'Predicting SVM #' + str(i) + ' for user ' + USERS[i]
    m = svm_load_model('model/user' + str(i) + '.model')
    p_label, p_acc, p_val = svm_predict(y, mfccFeatures, m)
    results['rec' + str(i)] = {'Name': USERS[i], 'Accuracy (%)': p_acc[0]}
   
    i = i+1
    
  print '************************************************************'
  print 'Predictions Complete'
  return results
  
  