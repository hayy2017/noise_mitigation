#!/usr/bin/python

############
# Description: script for preprocessing data and preparing the data stream
# Author: Heike Adel
# Year: 2016
###########

import sys
import os
import time
import numpy
from readData import readThreeContextsPlusEntityPairNumInstances, readEntityvectors, readEntityTuples, readEntitytypes, readConfig, readWordvectors
import copy

import theano

import cPickle

if len(sys.argv) != 2:
  print "please pass the config file as parameters"
  exit(0)

time1 = time.time()

configfile = sys.argv[1]
config = readConfig(configfile)

trainfile = config["train"]
print "trainfile ", trainfile
devfile = config["dev"]
print "devfile ", devfile
testfile = config["test"]
print "testfile ", testfile
wordvectorfile = config["wordvectors"]
print "wordvectorfile ", wordvectorfile
doEntityTypes = False
if "entitytypesTrain" in config:
  entitytypefileTrain = config["entitytypesTrain"]
  entitytypefileDev = config["entitytypesDev"]
  entitytypefileTest = config["entitytypesTest"]
  print "entity features: entity predictions"
  doEntityTypes = True
else:
  print "no entity types"
filename = config["file"]
print "filename for storing data ", filename
filtersize = [1,int(config["filtersize"])]
contextsize = int(config["contextsize"])
print "contextsize ", contextsize

if contextsize < filtersize[1]:
  print "INFO: setting filtersize to ", contextsize
  filtersize[1] = contextsize
print "filtersize ", filtersize

slot2ind = {}
slot2ind["government.government_agency.jurisdiction"] = 1
slot2ind["government.us_president.vice_president"] = 2
slot2ind["location.location.containedby"] = 3
slot2ind["organization.organization_founder.organizations_founded"] = 4
slot2ind["organization.organization.place_founded"] = 5
slot2ind["people.deceased_person.place_of_death"] = 6
slot2ind["people.person.children"] = 7
slot2ind["people.person.nationality"] = 8
slot2ind["people.person.place_of_birth"] = 9
slot2ind["people.person.religion"] = 10

time1 = time.time()

# reading word vectors
wordvectors, vectorsize = readWordvectors(wordvectorfile)

representationsize = vectorsize + 1

# read entity features
if doEntityTypes:
  print "reading entity type predictions"
  entities1Train, entities2Train, entityrepresentationsize = readEntitytypes(entitytypefileTrain)
  entities1Dev, entities2Dev, entityrepresentationsize = readEntitytypes(entitytypefileDev)
  entities1Test, entities2Test, entityrepresentationsize = readEntitytypes(entitytypefileTest)
  print len(entities1Train)
  print len(entities2Train)
  print len(entities1Dev)
  print len(entities2Dev)
  print len(entities1Test)
  print len(entities2Test)

def getMatrixForContext(context, vectorsize, contextsize, cap):
  global representationsize
  matrix = numpy.zeros(shape = (representationsize, contextsize))
  i = 0

  nextIndex = 0

  while i < len(context):
    word = context[i]
    nextIndex = 0
    # current word
    if word != "<empty>":
      if not word in wordvectors:
        word = "<unk>"
      curVector = wordvectors[word]
      for j in range(0, vectorsize):
        if j > len(curVector):
          print "ERROR: mismatch in word vector lengths: " + str(len(curVector)) + " vs " + vectorsize
          exit()
        elem = float(curVector[j])
        matrix[j + nextIndex, i] = elem
    nextIndex += vectorsize

    # capitalization feature
    matrix[nextIndex, i] = float(cap[i])

    i += 1

  return matrix

def padForConv(context, thisCap, curLength, filtersize, contextsize):
  contextNew = copy.deepcopy(context)
  thisCapNew = copy.deepcopy(thisCap)
  for dwin in range(filtersize / 2):
    contextNew.insert(0, "PADDING")
    contextNew.insert(curLength + dwin + 1, "PADDING")
    thisCapNew.insert(0, 0)
    thisCapNew.insert(curLength + dwin + 1, 0)
  contextNew = contextNew[0:contextsize]
  thisCapNew = thisCapNew[0:contextsize]
  while not contextNew[-filtersize/2 + 1] in ["PADDING", "<empty>"]:
    contextNew.pop(contextsize / 2)
    thisCapNew.pop(contextsize / 2)
    contextNew.append("PADDING")
    thisCapNew.append(0)
  return contextNew, thisCapNew

# read train file
inputListTrain_a, inputListTrain_b, inputListTrain_c, capTrain_a, capTrain_b, capTrain_c, lengthListTrain_a, lengthListTrain_b, lengthListTrain_c, nameBeforeFillerListTrain, resultVectorTrain, entityPairNumInstancesTrain = readThreeContextsPlusEntityPairNumInstances(trainfile, contextsize, slot2ind)
print "finished reading train"

numSamples = len(inputListTrain_a)

if numSamples == 0:
  print "no train examples for this slot: no training possible"
  exit()
inputs = [inputListTrain_a, inputListTrain_b, inputListTrain_c]
caps = [capTrain_a, capTrain_b, capTrain_c]
lenghts = [lengthListTrain_a, lengthListTrain_b, lengthListTrain_c]

inputMatrixTrain_a = numpy.empty(shape = (numSamples, representationsize * contextsize))
inputMatrixTrain_b = numpy.empty(shape = (numSamples, representationsize * contextsize))
inputMatrixTrain_c = numpy.empty(shape = (numSamples, representationsize * contextsize))
matrices = [inputMatrixTrain_a, inputMatrixTrain_b, inputMatrixTrain_c]
for sample in range(0, numSamples):
  for i,c,l,m in zip(inputs, caps, lenghts, matrices):
    context = i[sample]
    curLength = l[sample]
    thisCap = c[sample]
    context, thisCap = padForConv(context, thisCap, curLength, filtersize[1], contextsize)
    matrix = getMatrixForContext(context, vectorsize, contextsize, thisCap)
    matrix = numpy.reshape(matrix, representationsize * contextsize)
    m[sample,:] = matrix
print "finished processing train data"

# read dev file
inputListDev_a, inputListDev_b, inputListDev_c, capDev_a, capDev_b, capDev_c, lengthListDev_a, lengthListDev_b, lengthListDev_c, nameBeforeFillerListDev, resultVectorDev, entityPairNumInstancesDev = readThreeContextsPlusEntityPairNumInstances(devfile, contextsize, slot2ind)
print "finished reading dev"

numSamplesDev = len(inputListDev_a)
if numSamplesDev == 0:
  print "no dev examples for this slot: no training possible"
  exit()
inputs = [inputListDev_a, inputListDev_b, inputListDev_c]
caps = [capDev_a, capDev_b, capDev_c]
lengths = [lengthListDev_a, lengthListDev_b, lengthListDev_c]

inputMatrixDev_a = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
inputMatrixDev_b = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
inputMatrixDev_c = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
matrices = [inputMatrixDev_a, inputMatrixDev_b, inputMatrixDev_c]
for sample in range(0, numSamplesDev):
  for i,c,l,m in zip(inputs, caps, lengths, matrices):
    context = i[sample]
    curLength = l[sample]
    thisCap = c[sample]
    context, thisCap = padForConv(context, thisCap, curLength, filtersize[1], contextsize)
    matrix = getMatrixForContext(context, vectorsize, contextsize, thisCap)
    matrix = numpy.reshape(matrix, representationsize * contextsize)
    m[sample,:] = matrix
print "finished processing dev data"

# read test file
inputListTest_a, inputListTest_b, inputListTest_c, capTest_a, capTest_b, capTest_c, lengthListTest_a, lengthListTest_b, lengthListTest_c, nameBeforeFillerListTest, resultVectorTest, entityPairNumInstancesTest = readThreeContextsPlusEntityPairNumInstances(testfile, contextsize, slot2ind)
print "finished reading test"

numSamplesTest = len(inputListTest_a)
if numSamplesTest == 0:
  print "no test examples: no training possible"
  exit()
inputs = [inputListTest_a, inputListTest_b, inputListTest_c]
caps = [capTest_a, capTest_b, capTest_c]
lengths = [lengthListTest_a, lengthListTest_b, lengthListTest_c]

inputMatrixTest_a = numpy.empty(shape = (numSamplesTest, representationsize * contextsize))
inputMatrixTest_b = numpy.empty(shape = (numSamplesTest, representationsize * contextsize))
inputMatrixTest_c = numpy.empty(shape = (numSamplesTest, representationsize * contextsize))
matrices = [inputMatrixTest_a, inputMatrixTest_b, inputMatrixTest_c]
for sample in range(0, numSamplesTest):
  for i,c,l,m in zip(inputs, caps, lengths, matrices):
    context = i[sample]
    curLength = l[sample]
    thisCap = c[sample]
    context, thisCap = padForConv(context, thisCap, curLength, filtersize[1], contextsize)
    matrix = getMatrixForContext(context, vectorsize, contextsize, thisCap)
    matrix = numpy.reshape(matrix, representationsize * contextsize)
    m[sample,:] = matrix
print "finished processing test data"

time2 = time.time()
print "time for reading data: " + str(time2 - time1)

time1 = time.time()
dt = theano.config.floatX

################ FUEL #################
import h5py
from fuel.datasets.hdf5 import H5PYDataset

f = h5py.File(filename, mode='w')

features_a = f.create_dataset('xa', (numSamples + numSamplesDev + numSamplesTest, representationsize * contextsize), compression='gzip', dtype = dt)
features_b = f.create_dataset('xb', (numSamples + numSamplesDev + numSamplesTest, representationsize * contextsize), compression='gzip', dtype = dt)
features_c = f.create_dataset('xc', (numSamples + numSamplesDev + numSamplesTest, representationsize * contextsize), compression='gzip', dtype = dt)
if doEntityTypes:
  ent1 = f.create_dataset('ent1', (numSamples + numSamplesDev + numSamplesTest, entityrepresentationsize), dtype = dt, compression='gzip')
  ent2 = f.create_dataset('ent2', (numSamples + numSamplesDev + numSamplesTest, entityrepresentationsize), dtype = dt, compression='gzip')

targets = f.create_dataset('y', (numSamples + numSamplesDev + numSamplesTest, 1), dtype=numpy.dtype(numpy.int32), compression='gzip')
features_a[...] = numpy.vstack([inputMatrixTrain_a, inputMatrixDev_a, inputMatrixTest_a])
features_b[...] = numpy.vstack([inputMatrixTrain_b, inputMatrixDev_b, inputMatrixTest_b])
features_c[...] = numpy.vstack([inputMatrixTrain_c, inputMatrixDev_c, inputMatrixTest_c])
if doEntityTypes:
  ent1[...] = numpy.vstack([entities1Train, entities1Dev, entities1Test])
  ent2[...] = numpy.vstack([entities2Train, entities2Dev, entities2Test])
print "finished creating context features"
targets[...] = numpy.array(resultVectorTrain + resultVectorDev + resultVectorTest).reshape(numSamples + numSamplesDev + numSamplesTest, 1)

features_a.dims[0].label='batch'
features_a.dims[1].label='feature_a'
features_b.dims[0].label='batch'
features_b.dims[1].label='feature_b'
features_c.dims[0].label='batch'
features_c.dims[1].label='feature_c'
targets.dims[0].label='batch'
targets.dims[1].label='label'
if doEntityTypes:
  ent1.dims[0].label='batch'
  ent1.dims[1].label='entity1'
  ent2.dims[0].label='batch'
  ent2.dims[1].label='entity2'

if doEntityTypes:
  split_dict = {'train' : {'xa':(0,numSamples), 'xb':(0,numSamples), 'xc':(0,numSamples), 
    'ent1':(0,numSamples), 'ent2':(0,numSamples),
    'y':(0,numSamples)}, 
    'dev' : {'xa':(numSamples,numSamples+numSamplesDev), 'xb':(numSamples,numSamples+numSamplesDev), 'xc':(numSamples,numSamples+numSamplesDev), 
    'ent1':(numSamples, numSamples+numSamplesDev), 'ent2':(numSamples, numSamples+numSamplesDev),
    'y':(numSamples, numSamples+numSamplesDev)},
    'test' : {'xa':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'xb':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'xc':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 
    'ent1':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'ent2':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest),
    'y':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest)}}
else:
  split_dict = {'train' : {'xa':(0,numSamples), 'xb':(0,numSamples), 'xc':(0,numSamples), 'y':(0,numSamples)},
    'dev' : {'xa':(numSamples,numSamples+numSamplesDev), 'xb':(numSamples,numSamples+numSamplesDev), 'xc':(numSamples,numSamples+numSamplesDev), 'y':(numSamples, numSamples+numSamplesDev)},
    'test' : {'xa':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'xb':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'xc':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest), 'y':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()
print "finished writing H5PY file"

entityPairNumInstancesTrainNP = numpy.array(entityPairNumInstancesTrain)
entityPairNumInstancesDevNP = numpy.array(entityPairNumInstancesDev)
entityPairNumInstancesTestNP = numpy.array(entityPairNumInstancesTest)
entfile = open(filename + ".entities", 'wb')
cPickle.dump(entityPairNumInstancesTrainNP, entfile, -1)
cPickle.dump(entityPairNumInstancesDevNP, entfile, -1)
cPickle.dump(entityPairNumInstancesTestNP, entfile, -1)
entfile.close() 
print "finished writing entities file"
