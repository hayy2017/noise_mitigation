#!/usr/bin/python

############
# Description: Functions for reading config and data files
# Author: Heike Adel
# Year: 2016
###########

import gzip
import sys
import string
import re
import numpy

def readConfig(configfile):
  config = {}
  # read config file
  f = open(configfile, 'r')
  for line in f:
    if "#" == line[0]:
      continue # skip commentars
    line = line.strip()
    parts = line.split('=')
    name = parts[0]
    value = parts[1]
    config[name] = value
  f.close()
  return config

def readWordvectors(wordvectorfile, isWord2vec = True, readVocab = False, readLc = False):
  wordvectors = {}
  vectorsize = 0
  vocab = []
  f = open(wordvectorfile, 'r')
  count = 0
  for line in f:
    if isWord2vec:
      if count == 0:
        count += 1
        continue
    parts = line.split()
    if readLc:
      word = string.lower(parts[0])
    else:
      word = parts[0]
    parts.pop(0)
    wordvectors[word] = parts
    vectorsize = len(parts)
    vocab.append(word)
  f.close()
  if readVocab:
    return [wordvectors, vectorsize, vocab]
  else:
    return [wordvectors, vectorsize]

def readEntityvectors(entityvectorfile):
  entVec, vectorsize = readWordvectorsBasic(entityvectorfile, False, False, False)
  entVec2 = {}
  for e in entVec:
    e2 = re.sub(r'^\/m\/', 'm.', e)
    entVec2[e2] = entVec[e]
  return entVec2, vectorsize


def getArgumentIndices(contextWords):
    # improved index computation for splitting:
    # get all occurrences of <name> and <filler> and split where they are closest to each other
    # (idea: no <name> or <filler> tag in the middle context: keep middle context clean)
    fillerIndices = [i for i, x in enumerate(contextWords) if x == "<filler>"]
    nameIndices = [i for i, x in enumerate(contextWords) if x == "<name>"]
    fillerInd = -1
    nameInd = -1
    distanceNameFiller = len(contextWords)
    for fi in fillerIndices:
      for ni in nameIndices:
        distance = abs(ni - fi)
        if distance < distanceNameFiller:
          distanceNameFiller = distance
          nameInd = ni
          fillerInd = fi
    minIndex = 0
    maxIndex = 0
    nameBeforeFiller = 0
    if fillerInd == -1 or nameInd == -1:
      print "ERROR: no name or filler in " + str(contextWords)
      exit()
    if fillerInd < nameInd:
      nameBeforeFiller = 0
      minIndex = fillerInd
      maxIndex = nameInd
    else:
      nameBeforeFiller = 1
      maxIndex = fillerInd
      minIndex = nameInd
    return minIndex, maxIndex, nameBeforeFiller

def readEntityTuples(filename, is2015format):
  resultList = []
  if ".gz" in filename:
    f = gzip.open(filename, 'r')
  else:
    f = open(filename, 'r')
  for line in f:
    line = line.strip()
    if is2015format:
      parts = line.split(' :: ')
      e1 = parts[3]
      e2 = parts[4]
    else:
      parts = line.split(' : ')
      e1 = parts[2]
      e2 = parts[3]
    resultList.append((e1,e2))
  return resultList

def readEntitytypes(inputfile):
  entities = numpy.load(inputfile)
  vectorsize = entities.shape[1]
  e1 = entities[::2]
  e2 = entities[1::2]
  return e1, e2, vectorsize


def readFilePlusEntityPairNumInstancesClueweb(filename, is2015format, slot2ind = {}, entitiesInContext = True, averageEntityWords = False):
  if averageEntityWords:
    # read entity file
    entityfilename = re.sub(r'SFformat$', 'entityWords', filename)
    entities = []
    f = open(entityfilename, 'r')
    for line in f:
      line = line.strip()
      entities.append(line.split('\t'))
    f.close()
  inputList_a = []
  inputList_b = []
  inputList_c = []
  resultVector = []
  nameBeforeFillerList = []
  ePairs = []
  eCount = 0
  prevEPair = ("", "")
  if ".gz" in filename:
    f = gzip.open(filename, 'r')
  else:
    f = open(filename, 'r')
  curIndex = 0
  for line in f:
    line = line.strip()
    if is2015format:
      parts = line.split(' :: ')
      posNeg = parts[1]
      slot = parts[2]
      contextWords = " :: ".join(parts[5:]).split()
      ePair = (parts[3], parts[4])
    else:
      parts = line.split(' : ')
      posNeg = parts[0]
      slot = parts[1]
      ePair = (parts[2], parts[3])
      contextWords = " : ".join(parts[4:])
      contextWords = re.sub(r'\/m\/(.*?)\/(.*?)\/', '/m/\\1 ', contextWords)
      contextWords = re.sub(r' +', ' ', contextWords)
      contextWords = contextWords.split()
    if not "<name>" in contextWords or not "<filler>" in contextWords:
      sys.stderr.write("skipping example because there is no <name> and/or <filler> tag: " + " ".join(contextWords) + "\n")
      continue # skip example
    minIndex, maxIndex, nameBeforeFiller = getArgumentIndices(contextWords)
    nameBeforeFillerList.append(nameBeforeFiller)
    if ePair == prevEPair:
      eCount += 1
    else:
      if prevEPair != ("", ""):
        ePairs.append(eCount)
      eCount = 1
      prevEPair = ePair
    if averageEntityWords:
      leftContext = " ".join(contextWords[:minIndex + 1])
      middleContext = " ".join(contextWords[minIndex : maxIndex + 1])
      rightContext = " ".join(contextWords[maxIndex:])
      e1, e2 = entities[curIndex]
      leftContext = re.sub(r'\<name\>', e1, leftContext)
      leftContext = re.sub(r'\<filler\>', e2, leftContext)
      middleContext = re.sub(r'\<name\>', e1, middleContext)
      middleContext = re.sub(r'\<filler\>', e2, middleContext)
      rightContext = re.sub(r'\<name\>', e1, rightContext)
      rightContext = re.sub(r'\<filler\>', e2, rightContext)
      leftContext = leftContext.split()
      middleContext = middleContext.split()
      rightContext = rightContext.split()
    elif entitiesInContext:
      leftContext = " ".join(contextWords[:minIndex + 1])
      middleContext = " ".join(contextWords[minIndex : maxIndex + 1])
      rightContext = " ".join(contextWords[maxIndex:])
      e1 = re.sub(r'm\.', '/m/', ePair[0])
      e2 = re.sub(r'm\.', '/m/', ePair[1])
      leftContext = re.sub(r'\<name\>', e1, leftContext)
      leftContext = re.sub(r'\<filler\>', e2, leftContext)
      middleContext = re.sub(r'\<name\>', e1, middleContext)
      middleContext = re.sub(r'\<filler\>', e2, middleContext)
      rightContext = re.sub(r'\<name\>', e1, rightContext)
      rightContext = re.sub(r'\<filler\>', e2, rightContext)
      leftContext = leftContext.split()
      middleContext = middleContext.split()
      rightContext = rightContext.split()
    else:
      leftContext = contextWords[:minIndex]
      middleContext = contextWords[minIndex + 1 : maxIndex]
      rightContext = contextWords[maxIndex + 1:]
    inputList_a.append(leftContext)
    inputList_b.append(middleContext)
    inputList_c.append(rightContext)
    if len(slot2ind.keys()) == 0:
      if posNeg == '+':
        resultVector.append(1)
      else:
        resultVector.append(0)
    else:
      if slot in slot2ind:
        resultVector.append(slot2ind[slot])
      else:
        resultVector.append(0) # negative relation
    curIndex += 1
  ePairs.append(eCount)
  return inputList_a, inputList_b, inputList_c, resultVector, nameBeforeFillerList, ePairs

def padRight(context, newSize):
  while len(context) < newSize:
    context.append('<empty>')
  return context

def popLeft(context, newSize):
  while len(context) > newSize:
    context.pop(0)
  return context

def popMiddle(context, newSize):
  while len(context) > newSize:
    indexToRemove = (len(context) - 1) / 2
    context.pop(indexToRemove)
  return context

def popRight(context, newSize):
  while len(context) > newSize:
    context.pop(-1)
  return context


def readThreeContextsAndLengthPlusEntityPairNumInstances(filename, contextsize, slot2ind = {}, entitiesInContext = True, averageEntityWords = False):
  left, middle, right, resultVector, nameBeforeFillerList, entityPairs = readFilePlusEntityPairNumInstancesClueweb(filename, False, slot2ind, entitiesInContext, averageEntityWords)

  inputList_a = []
  inputList_b = []
  inputList_c = []
  lengthList_a = []
  lengthList_b = []
  lengthList_c = []

  index = 0
  for contextWords_a, contextWords_b, contextWords_c in zip(left, middle, right):

    myLength_a = min(contextsize, len(contextWords_a))
    myLength_a = max(1, myLength_a)
    myLength_b = min(contextsize, len(contextWords_b))
    myLength_b = max(1, myLength_b)
    myLength_c = min(contextsize, len(contextWords_c))
    myLength_c = max(1, myLength_c)

    # adjust left context
    contextWords_a = padRight(contextWords_a, contextsize)
    contextWords_a = popLeft(contextWords_a, contextsize)
    # adjust middle context
    contextWords_b = padRight(contextWords_b, contextsize)
    contextWords_b = popMiddle(contextWords_b, contextsize)
    # adjust right context
    contextWords_c = padRight(contextWords_c, contextsize)
    contextWords_c = popRight(contextWords_c, contextsize)

    inputList_a.append(contextWords_a)
    lengthList_a.append(myLength_a)
    inputList_b.append(contextWords_b)
    lengthList_b.append(myLength_b)
    inputList_c.append(contextWords_c)
    lengthList_c.append(myLength_c)

  return [inputList_a, inputList_b, inputList_c, lengthList_a, lengthList_b, lengthList_c, nameBeforeFillerList, resultVector, entityPairs]

def getLowerCasedVersion(contexts):
  lc_contexts = []
  cap_contexts = []
  for c in contexts:
    lc_a = []
    cap_a = []
    for a in c:
      tmp_a = []
      tmp_cap_a = []
      for part in a:
        if part[0].isupper() and part != "PADDING":
          tmp_cap_a.append(1)
        else:
          tmp_cap_a.append(0)
        if part == "PADDING":
          tmp_a.append(part)
        else:
          tmp_a.append(string.lower(part))
      lc_a.append(tmp_a)
      cap_a.append(tmp_cap_a)
    lc_contexts.append(lc_a)
    cap_contexts.append(cap_a)
  return lc_contexts, cap_contexts

def readThreeContextsPlusEntityPairNumInstances(filename, contextsize, slot2ind = {}, entitiesInContext = True, averageEntityWords = False, lc = True):
  normal = readThreeContextsAndLengthPlusEntityPairNumInstances(filename, contextsize, slot2ind, entitiesInContext, averageEntityWords)
  if lc:
    lc_contexts, cap_contexts = getLowerCasedVersion(normal[0:3])
    return lc_contexts[0:3] + cap_contexts[0:3] + normal[3:]
  else:
    return normal

def getEntityPairNumInstances(filename, is2015format = False):
  if ".gz" in filename:
    f = gzip.open(filename, 'r')
  else:
    f = open(filename, 'r')
  countList = []
  previousPair = ("","")
  count = 0
  for line in f:
    line = line.strip()
    if is2015format:
      parts = line.split(' :: ')
      e1 = parts[3]
      e2 = parts[4]
    else:
      parts = line.split(' : ')
      e1 = parts[2]
      e2 = parts[3]
  if (e1,e2) == previousPair:
    count += 1
  else:
    if count != 0: # count == 0: we are in first line
      countList.append(count)
    count = 1
    previousPair = (e1,e2)
  countList.append(count) # last entity pair
  f.close()
  return countList

def getEntityPairs(filename, is2015format = False):
  if ".gz" in filename:
    f = gzip.open(filename, 'r')
  else:
    f = open(filename, 'r')
  countList = []
  ent2index = {}
  count = 0
  for line in f:
    line = line.strip()
    if is2015format:
      parts = line.split(' :: ')
      e1 = parts[3]
      e2 = parts[4]
    else:
      parts = line.split(' : ')
      e1 = parts[2]
      e2 = parts[3]
    if (e1,e2) in ent2index:
      countList.append(ent2index[(e1,e2)])
    else:
      count += 1
      ent2index[(e1,e2)] = count
      countList.append(count)
  f.close()
  return countList

