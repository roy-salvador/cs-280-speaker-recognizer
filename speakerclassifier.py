import pyaudio
import numpy
import scipy.io.wavfile as wav
from features import mfcc
from features import ssc
import math
from svmutil import *
import datetime
import random
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

# SVM Training Data
USERS = []
RAW_TRAINING_FEATURES = []
RAW_TRAINING_LABELS = []

# Initialization
def init() :
  global USERS
  global RAW_TRAINING_FEATURES
  global RAW_TRAINING_LABELS
  
  # Load Users
  print 'Initializing saved users'
  usersDB = open('model\users.csv', "r")
  for user in usersDB :
    USERS.append(user.strip())
  usersDB.close()
  
  # Load Raw Training Data
  print 'Initializing saved training data'
  trainFile = open("model/raw_features_training.csv", "r")
  for line in trainFile:
    j=0
    featuresVector = []
    for value in line.strip().split(',') :
      if j==0 :
        RAW_TRAINING_LABELS.append(int(value))
      else :
        featuresVector.append(float(value)) 
      j = j+1
    RAW_TRAINING_FEATURES.append(featuresVector)	
  trainFile.close()
    

# Extracts the MFCC Feature vectors of the audio file
def extract_features(audioFile) :
  print 'Extracting Features of ' + audioFile
  audioFeatures = []
  
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
      # Get feature of the current frame
      featureVectors = [] #mfcc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH) + ssc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH)
      mfccFeatures = [] 
      sscFeatures = [] 
      mfccFeatures = mfcc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH).tolist()
      sscFeatures = ssc(windowedFrame, samplerate=rate, winlen=FRAME_LENGTH).tolist()
      j=0
      while j<len(mfccFeatures) :
        featureVectors.append(mfccFeatures[j] + sscFeatures[j])
        j=j+1
      
      #audioFeatures = audioFeatures + featureVectors.tolist() 
      audioFeatures = audioFeatures + featureVectors
      
    #print str(i) + ' ' + str(numpy.mean(numpy.absolute(windowedFrame)))
    i = i + overlap
 
  print 'There are ' + str(len(audioFeatures)) + ' extracted MFCC feature vectors.'
  return audioFeatures

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

# Returns a sampled dataset for the user including all the user's original data + sampled data from other users
#   1 tagged as MFCC feature of the user
#  -1 tagged as MFCC feature is not the user's
def getTrainingDataForUser(userID):
  labels = []
  attributes =[]
  negative_class = []
  
  i=0
  while i<len(RAW_TRAINING_FEATURES) :
    if userID == RAW_TRAINING_LABELS[i]:
      labels.append(1)
      attributes.append(RAW_TRAINING_FEATURES[i])
    else :
      negative_class.append(RAW_TRAINING_FEATURES[i])
    i=i+1
  user_instance_count = len(labels)  
    
  while (len(labels) <  user_instance_count*2) and (len(negative_class) > 0) :
    random_index = int(math.floor(len(negative_class)*random.random()))
    
    labels.append(-1)
    attributes.append(negative_class[random_index])
    del negative_class[random_index]
  
  return labels, attributes

# Scale the raw feature points so that each "column" is
# normalized to the same scale
# Linear stretch from lowest value = -1 to highest value = 1
def scaleFeatures(audioFeatures) :
  high = 1.0
  low = -1.0
  
  # use scaling factors of the Training set
  mins = numpy.min(RAW_TRAINING_FEATURES, axis=0)
  maxs = numpy.max(RAW_TRAINING_FEATURES, axis=0)
  rng = maxs - mins
  
  scaled_points = high - (((high - low) * (maxs - audioFeatures)) / rng)
  return scaled_points.tolist()
  
  
  
# Trains all the user SVMS with the new training data of the new user
def train_user_svms(name, audioFile) :
  global RAW_TRAINING_FEATURES
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
  audioFeatures = extract_features(audioFile)
  RAW_TRAINING_FEATURES = RAW_TRAINING_FEATURES + audioFeatures
  i=0
  while i < len(audioFeatures) :
    RAW_TRAINING_LABELS.append(userID)
    i=i+1
  # Persist training data
  print 'Saving obtained training data'
  featuresFile = open("model/raw_features_training.csv", "w")
  i=0
  while i<len(RAW_TRAINING_LABELS) :
    featuresFile.write(str(RAW_TRAINING_LABELS[i]) + ',' + str(RAW_TRAINING_FEATURES[i]).lstrip('[').rstrip(']') + '\n')
    i = i+1
  featuresFile.close()
  
  # Scale the training data first 
  #scaledMFCC = scaleFeatures(RAW_TRAINING_FEATURES)

  # Train an SVM for each user
  
  i=0
  while i < len(USERS) :
    print '************************************************************'
    print 'Training SVM #' + str(i) + ' for user ' + USERS[i]
    print  'Start Time: ' + str(datetime.datetime.now()) 
    labelVector, attribs = getTrainingDataForUser(i) #getLabelVectorForUser(i)
    
    # Compute class weights for unbalanced data
    #positiveWeight =  1.0*labelVector.count(1)/len(labelVector)
    #negativeWeight = 1.0*labelVector.count(-1)/len(labelVector)
    
    prob = svm_problem(labelVector, scaleFeatures(attribs))
    param = svm_parameter('-t 2 -c 32.0 -g 0.76923 ') #-w1 ' + str(negativeWeight) + ' -w-1 ' + str(positiveWeight) )  #use RBF kernel
    m = svm_train(prob, param)
    svm_save_model('model/user' + str(i) + '.model', m)
    i = i+1
    print  'End Time: ' + str(datetime.datetime.now()) 
    
  print '************************************************************'
  print 'Training Complete'
 
# Returns the SVM prediction accuracies in the dict format the tktable can use 
def classify_audio(audioFile) : 
  results = {}
  print '============================================================================================='
  print 'Predicting which user is speaking in the audio file'
  
  # Extract Features and initialize label vector
  audioFeatures = extract_features(audioFile)
  y = []
  for mf in audioFeatures:
    y.append(1)   # assume features vector belongs to the user
   
  # Scale the test data using scaling factors used in training data
  scaledAudioFeatures = scaleFeatures(audioFeatures)
  
  # Get accuracy for every user
  i=0
  while i < len(USERS) :
    print '************************************************************'
    print 'Predicting SVM #' + str(i) + ' for user ' + USERS[i]
    m = svm_load_model('model/user' + str(i) + '.model')
    p_label, p_acc, p_val = svm_predict(y, scaledAudioFeatures, m)
    results['rec' + str(i)] = {'Name': USERS[i], 'Score (%)': round(p_acc[0],2)}
   
    i = i+1
    
  print '************************************************************'
  print 'Predictions Complete'
  return results
  
  