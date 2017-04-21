#!/usr/bin/python

############
# Description: script for loading and evaluating a previously trained model in terms of PR curve
# Author: Heike Adel
# Year: 2016
###########

import sys
import time
import numpy
import theano
import theano.tensor as T
from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer
import cPickle
from blocks.roles import add_role, WEIGHT

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from blocks_fuel_classes import MultiInstanceScheme, MultiInstanceSchemeShuffled, load

from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks_fuel_classes import GetPRcurve

from readData import readConfig

if len(sys.argv) != 2:
  print "please pass the config file as parameters"
  exit(0)

time1 = time.time()

configfile = sys.argv[1]
config = readConfig(configfile)

datafile = config["file"]
print "datafile ", datafile
vectorsize = int(config["vectorsize"])
print "vectorsize ", vectorsize
entityrepresentationsize = int(config["entitysize"])
print "entity representation size ", entityrepresentationsize
networkfile = config["net"]
print "networkfile ", networkfile
learning_rate = float(config["lrate"])
print "learning rate ", learning_rate
batch_size = int(config["batchsize"])
print "batch size ", batch_size
filtersize = [1,int(config["filtersize"])]
nkerns = [int(config["nkerns"])]
print "nkerns ", nkerns
pool = [1, int(config["kmax"])]
contextsize = int(config["contextsize"])
print "contextsize ", contextsize
numClasses = int(config["numClasses"]) + 1 # plus 1 negative class ($NA)
print "number of classes: " + str(numClasses)
myLambda1 = 0
if "lambda1" in config:
  myLambda1 = float(config["lambda1"])
myLambda2 = 0
if "lambda2" in config:
  myLambda2 = float(config["lambda2"])
print "lambda1 ", myLambda1
print "lambda2 ", myLambda2
useHiddenLayer = True
if "noHidden" in config:
  useHiddenLayer = False
  print "using no hidden layer"
else:
  hiddenunits = int(config["hidden"])
  print "hidden units ", hiddenunits
useEntityTypes = True
if "noTypes" in config:
  useEntityTypes = False
useHiddenForTypes = False
if "hiddentype" in config:
  hiddenTypeUnits = int(config["hiddentype"])
  useHiddenForTypes = True
  print "hidden type units ", hiddenTypeUnits

iterationSeed = -1
if "iterationSeed" in config:
  iterationSeed = int(config["iterationSeed"])
  print "using " + str(iterationSeed) + " as seed for iteration scheme"

if contextsize < filtersize[1]:
  print "INFO: setting filtersize to ", contextsize
  filtersize[1] = contextsize
print "filtersize ", filtersize

sizeAfterConv = contextsize - filtersize[1] + 1

sizeAfterPooling = -1
if sizeAfterConv < pool[1]:
  print "INFO: setting poolsize to ", sizeAfterConv
  pool[1] = sizeAfterConv
if "kmax" in config or "variableLength":
  sizeAfterPooling = pool[1]
  print "kmax pooling: k = ", pool[1]
else:
  sizeAfterPooling = sizeAfterConv / pool[1]
  if sizeAfterConv % pool[1] != 0:
    sizeAfterPooling += 1
  print "traditional pooling: pool = ", pool[1]

time1 = time.time()

representationsize = vectorsize + 1

time2 = time.time()
print "time for reading data: " + str(time2 - time1)

# train network
curSeed = 23455
if "seed" in config:
  curSeed = int(config["seed"])
rng = numpy.random.RandomState(curSeed)
seed = rng.get_state()[1][0]
print "seed: " + str(seed)

time1 = time.time()
dt = theano.config.floatX

#### data stream ##############
# The names here (e.g. 'name1') need to match the names of the variables which
#  are the roots of the computational graph for the cost.

train_set = H5PYDataset(datafile, which_sets = ('train',), load_in_memory = True)
dev_set = H5PYDataset(datafile, which_sets = ('dev',), load_in_memory = True)
test_set = H5PYDataset(datafile, which_sets = ('test',), load_in_memory = True)
bagF = open(datafile + ".entities")
bag_size_list_train = cPickle.load(bagF)
bag_size_list_dev = cPickle.load(bagF)
bag_size_list_test = cPickle.load(bagF)
bagF.close()
numSamplesTrain = train_set.num_examples
numSamplesDev = dev_set.num_examples
numSamplesTest = test_set.num_examples

print "got " + str(numSamplesTrain) + " train examples"
print "got " + str(numSamplesDev) + " dev examples"
print "got " + str(numSamplesTest) + " test examples"

data_stream = DataStream(train_set, iteration_scheme = MultiInstanceSchemeShuffled(train_set.num_examples, bag_size_list_train))
data_stream_dev = DataStream(dev_set, iteration_scheme=MultiInstanceScheme(dev_set.num_examples, bag_size_list_dev))
data_stream_test = DataStream(test_set, iteration_scheme=MultiInstanceScheme(test_set.num_examples, bag_size_list_test))

################################

# allocate symbolic variables for the data
xa = T.matrix('xa')   # the data is presented as rasterized images
xb = T.matrix('xb')
xc = T.matrix('xc')
y = T.imatrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
ent1 = T.matrix('ent1') # feature for entity 1 (e.g. embedding or type prediction)
ent2 = T.matrix('ent2') # feature for entity 2
ishape = [representationsize, contextsize]  # this is the size of context matrizes

time2 = time.time()
print "time for preparing data structures: " + str(time2 - time1)

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'
time1 = time.time()
# Reshape matrix of rasterized images of shape (batch_size,28*28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
#layer0_input = T.concatenate([xa.reshape((batch_size, 1, ishape[0], ishape[1])), xb.reshape((batch_size, 1, ishape[0], ishape[1])), xc.reshape((batch_size, 1, ishape[0], ishape[1]))], axis = 3)
layer0a_input = xa.reshape((xa.shape[0], 1, ishape[0], ishape[1]))
layer0b_input = xb.reshape((xb.shape[0], 1, ishape[0], ishape[1]))
layer0c_input = xc.reshape((xc.shape[0], 1, ishape[0], ishape[1]))

y = y.reshape((xa.shape[0], ))

# Construct the first convolutional pooling layer:
filter_shape = (nkerns[0], 1, representationsize, filtersize[1])
poolsize=(pool[0], pool[1])

fan_in = numpy.prod(filter_shape[1:])
fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
              numpy.prod(poolsize))

W_bound = numpy.sqrt(6. / (fan_in + fan_out))
# the convolution weight matrix
convW = theano.shared(numpy.asarray(
           rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
           dtype=theano.config.floatX), name='conv_W',
                               borrow=True)

# the bias is a 1D tensor -- one bias per output feature map
b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
convB = theano.shared(value=b_values, name='conv_b', borrow=True)

layer0a = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0a_input,
            image_shape=(xa.shape[0], 1, ishape[0], ishape[1]),
            filter_shape=filter_shape, poolsize=poolsize)
layer0b = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0b_input,
              image_shape=(xb.shape[0], 1, ishape[0], ishape[1]),
              filter_shape=filter_shape, poolsize=poolsize)
layer0c = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0c_input,
              image_shape=(xc.shape[0], 1, ishape[0], ishape[1]),
              filter_shape=filter_shape, poolsize=poolsize)
layer0flattened = T.concatenate([layer0a.output.flatten(2), layer0b.output.flatten(2), layer0c.output.flatten(2)], axis = 1)
#.reshape((batch_size, nkerns[0] * sizeAfterPooling))


if useEntityTypes:
  ent1 = ent1.flatten(2)  
  ent2 = ent2.flatten(2)

  if useHiddenForTypes:
    layer1a = HiddenLayer(rng, input=ent1, n_in=entityrepresentationsize, n_out=hiddenTypeUnits, activation=T.tanh, name="type_")
    layer1b = HiddenLayer(rng, input=ent2, n_in=entityrepresentationsize, n_out=hiddenTypeUnits, activation=T.tanh, W = layer1a.W, b = layer1a.b, name="type_")
    layer2_input = T.concatenate([layer0flattened, layer1a.output, layer1b.output], axis = 1)
    layer2_inputSize = nkerns[0] * sizeAfterPooling * 3 + 2 * hiddenTypeUnits
  else:
    layer2_input = T.concatenate([layer0flattened, ent1, ent2], axis = 1)
    layer2_inputSize = nkerns[0] * sizeAfterPooling * 3 + 2 * entityrepresentationsize
else:
  layer2_input = layer0flattened
  layer2_inputSize = nkerns[0] * sizeAfterPooling * 3

if useHiddenLayer:
  # construct a fully-connected sigmoidal layer
  layer2 = HiddenLayer(rng, input=layer2_input, n_in=layer2_inputSize,
                         n_out=hiddenunits, activation=T.tanh)
  # classify the values of the fully-connected sigmoidal layer
  layer3 = LogisticRegression(input=layer2.output, n_in=hiddenunits, n_out=numClasses)
else:
  # classify the values of the fully-connected sigmoidal layer
  layer3 = LogisticRegression(input=layer2_input, n_in=layer2_inputSize, n_out=numClasses)

# create a list of all model parameters to be fit by gradient descent
paramList = [layer3.params]
if useHiddenLayer:
  paramList.append(layer2.params)
if useHiddenForTypes:
  paramList.append(layer1a.params)
paramList.append(layer0a.params)

params = []
for p in paramList:
  for p_part in p:
    add_role(p_part, WEIGHT)
  params += p

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood_mi(y)

reg2 = 0
reg1 = 0
for p in paramList:
  reg2 += T.sum(p[0] ** 2)
  reg1 += T.sum(abs(p[0]))
cost += myLambda2 * reg2
cost += myLambda1 * reg1

cost.name = 'cost'

lr = T.scalar('lr', dt)

time2 = time.time()
print "time for building the model: " + str(time2 - time1)


######### load model and get results ##################

n_epochs = "15"
if "n_epochs" in config:
  n_epochs = config["n_epochs"]

finalNetworkfile = networkfile + "." + n_epochs

model = Model([cost])
f = open(finalNetworkfile)
old_main_loop = load(f)
f.close()
old_model = old_main_loop.model
model.set_parameter_values(old_model.get_parameter_values())

extensions = []
algorithm = None
extensions.append(GetPRcurve(layer3 = layer3, y = y, model = model, data_stream = data_stream_test, num_samples = len(bag_size_list_test), batch_size = bag_size_list_test, before_training=True))
my_loop = MainLoop(model=model,
                   data_stream=data_stream,
                   algorithm=algorithm,
                   extensions=extensions)
for extension in my_loop.extensions:
  extension.main_loop = my_loop
my_loop._run_extensions('before_training')

