'''
Created on Nov 1, 2015

@author: yy1
'''
import numpy, os, sys, logging, string, random
from _collections import defaultdict
import gzip, bz2
import multiprocessing
import theano
import math, yaml
from multiprocessing.pool import Pool
from collections import OrderedDict
import collections
from numpy import argmax
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')

random.seed(100000)

def str_to_bool(s):
    if s == 'True' or s == 'true':
        return True
    elif s == 'False' or s == 'false':
        return False
    else:
        return s
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(Pool):
    Process = NoDaemonProcess

def myopen(filename, mode='r'):
    """
    Open file. Use gzip or bzip2 if appropriate.
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)

    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename, mode)

    return open(filename, mode)

def loadConfig(configFile):
    if '.yaml' in configFile: 
        with open(configFile) as fp:
            return yaml.load(fp)

    config = {}
    f = open(configFile, 'r')
    for line in f:
        if "#" == line[0]:
            continue  # skip commentars
        line = line.strip()
        if line == '': continue
        parts = line.split('=')
        name = parts[0]
        value = parts[1]
        config[name] = value
    f.close()
    return config

def openloud(myname):
    myfile = open(myname, 'r')
    print 'now open:', myname
    return myfile
def readall(filename, loud=True):
    if loud:
        myfile = openloud(filename)
    else:
        myfile = open(filename, 'r')
    mystring = myfile.read()
    myfile.close()
    return mystring

def getlines(filename):
    fcontents = readall(filename)
    lines = string.split(fcontents, '\n')
    return lines

def getlinesm1(filename, loud=True):
    fcontents = readall(filename, loud)
    lines = string.split(fcontents, '\n')
    if len(string.strip(lines[-1])) == 0:
        lines = lines[:-1]
    return lines

def getentparts(ent_token):
    ent_parts = ent_token.split('##')
    parts = ent_parts[0].split('/')
    if len(parts) < 4:
        return (ent_parts[0], 'unk', ent_parts[1])
    mid = '/m/' + parts[2]
    tokens = parts[3].split('_')
    if len(ent_parts) < 2:
        print ent_token
    notabletype = ent_parts[1]
    return (mid, tokens, notabletype)

def convertTargetsToBinVec(other_types_ind, n_out):
    outvec = numpy.zeros(n_out, numpy.int32)
    for ind in other_types_ind:
        outvec[ind] = 1
    return outvec

def parsethesampleline(myline, type2ind):
    parts = myline.split('\t')
    myent = parts[1]
    notable_type = parts[2]
    if '(' in notable_type:
        notable_type = notable_type[0:notable_type.index('(')].strip()
    if notable_type not in type2ind:
        print 'type not found: ' + notable_type  
        assert 1
    ntindex = type2ind[notable_type]
    other_types = parts[3].split(',')
    types_ind = []
    for i in range(0,len(other_types)):
        if other_types[i] not in type2ind:
            continue
        types_ind.append(type2ind[other_types[i]])
    types_ind.append(ntindex)
    return (myent, ntindex, types_ind, parts[4].split(' '))
    
def getwindowcontext(tokens, myent, maxwindow):
    onewaysize = maxwindow / 2
    c = 0; entind = 0
    newtokens = []
    for word in tokens:
        if "/m/" in word:
            (mid, ent_words, notabletype) = getentparts(word)
            if mid == myent and entind == 0:
                entind = c
            newtokens.append(notabletype)
        else:
            newtokens.append(word)
        c += 1
    assert len(newtokens) == len(tokens) 
    thiscontext = ''
    for i in range(entind - onewaysize, entind + onewaysize + 1):
        if i < 0 or i >= len(newtokens):
            thiscontext += '<PAD>'
        else:
            thiscontext += newtokens[i]
        thiscontext += ' '
    return thiscontext.strip()  

def read_lines_data(myname, type2ind, maxwindow=10, targetlabel='nt', upto=-1):
    res_vec = []; res_vec_all = []
    contextlist = []
    tmp = getlinesm1(myname)
    print 'lines loaded from', myname
    print 'now building the context windows'
    mycount = -1
    for myline in tmp:
        mycount += 1
        if upto >= 0 and mycount >= upto: break
        (myent, nt, alltypes, tokens) = parsethesampleline(myline.strip(), type2ind)
        res_vec.append(nt)
        res_vec_all.append(convertTargetsToBinVec(alltypes, len(type2ind)))
        thiscontext = getwindowcontext(tokens, myent, maxwindow)
        contextlist.append(thiscontext.strip())
    return(contextlist, res_vec, res_vec_all)

def loadTargets(filename):
    # Filling type indexes from a type list file
    targetIndMap = dict()
    typefreq_traindev = dict()
    c = 0
    f = open(filename, 'r')
    for line in f:
        t = line.split('\t')[0].strip()
        targetIndMap[t] = c
        if len(line.split('\t')) < 3:
            typefreq = 0
        else:
            typefreq = line.split('\t')[2].strip()
        if typefreq == 'null':
            typefreq_traindev[t] = 0
        else:
            typefreq_traindev[t] = int(typefreq)
        c += 1
    return (targetIndMap, len(targetIndMap), typefreq_traindev) 

def yyreadwordvectors(filename, upto=-1):
    # reading word vectors
    print 'loading word vectors'
    wordvectors = defaultdict(list)
    count = 0
    with open(filename) as fp:
        vectorsize = int(fp.readline().split()[1])
        wordvectors['<UNK>'] = [0.001 for _ in range(vectorsize)]
        for line in fp:
            count += 1
            if upto != -1 and count > upto:
                break;
            parts = line.strip().split()
            word = parts[0].strip()
            parts.pop(0)
            wordvectors[word] = numpy.array([float(s) for s in parts])
    print "vector size is: " + str(vectorsize)
    print len(wordvectors)
    return (wordvectors, vectorsize)

def read_embeddings(fname, num=-1):
    """ Read word embeddings from file

    :param embedding_file:  Path to embedding file
    :type embedding_file:   str/unicode
    :param num:             Restrict number of embeddings to load
    :type num:              int
    :returns:               Mapping from words to their embedding
    :rtype:                 str -> numpy.array dict
    """
    # NOTE: This should be pretty efficient, since we're only reading the file
    #       line by line and are reading directly into memory-efficient numpy
    #       arrays.
    logger.info("loading word2vec vectors from: %s", fname)
    with open(fname) as fp:
        num_vecs, vec_size = (int(x) for x in fp.readline().strip().split())
        num_vecs += 2  # For <UNK>
        embeddings = numpy.zeros((num_vecs, vec_size), dtype='float32')
        embeddings[0,:] = 0.001
        word_to_idx = {'<UNK>': 0, '<PAD>': 1}
        for idx, line in enumerate(fp, start=2):
            if num != -1 and idx > num:
                break
            parts = line.strip().split()
            embeddings[idx,:] = [float(v) for v in parts[1:]]
            word_to_idx[parts[0]] = idx
    logger.info("number of words in word2vec: %d", len(word_to_idx))
    return embeddings, word_to_idx, vec_size

def buildtypevectorsmatrix(type2ind, wordvecs, vectorsize):
    m = numpy.zeros(shape=(len(type2ind), vectorsize))
    for t in type2ind:
        ind = type2ind[t]
        m[ind] = wordvecs[t]
    return m
    
def calcInsertMeanInputs(matrix, slotPosition, meanwindow):
    l = meanwindow / 2
    meanvec = numpy.mean(matrix[:,slotPosition-l:slotPosition+l+1], axis=1)
    matrix[:,slotPosition] = meanvec
#     for i in range(0, matrix.shape[0]):
#         matrix[i][slotPosition] = meanofrows[i] 
    return matrix, meanvec

def adeltheanomatrix_flexible(slotposition, vectorsize, contextlist, wordvectors, leftsize, rightsize, sum_window=10, insertsum=True):
    numvec = len(contextlist)
    sample = 0
    desired_window_size = leftsize + rightsize + 1
    myMatrix = numpy.zeros(shape=(numvec, vectorsize * desired_window_size))
    while True:
        if sample >= len(contextlist): break
        thiscontext = contextlist[sample]
        contextWords = thiscontext.split()
        assert len(contextWords) >= desired_window_size
        matrix = numpy.zeros(shape=(vectorsize, len(contextWords)))
        for i in range(0, len(contextWords)):
            if i == slotposition: continue
            word = contextWords[i]
            if word == '<PAD>': continue
            if word in wordvectors:
                curvec = wordvectors[word]
            else:
                curvec = wordvectors['<UNK>']
            matrix[:,i] = curvec
        if insertsum:
            (matrix, meanvec) = calcInsertMeanInputs(matrix, slotposition, sum_window)
        newlimitedmatrix = matrix[:,slotposition-leftsize:slotposition+rightsize+1]
        newlimitedmatrix = numpy.reshape(newlimitedmatrix, vectorsize * desired_window_size)
        myMatrix[sample, :] = newlimitedmatrix
        sample += 1
    return myMatrix

def load_features(typematrix, slotposition, vectorsize, contextlist, wordvectors, leftsize, rightsize, sum_window=10):
    numvec = len(contextlist)
    sample = 0
    desired_window_size = leftsize + rightsize + 1
    myMatrix = numpy.zeros(shape=(numvec, vectorsize * desired_window_size + len(typematrix))) # +1 for the cosine sim vector
    while True:
        if sample >= len(contextlist): break
        thiscontext = contextlist[sample]
        contextWords = thiscontext.split()
        assert len(contextWords) >= desired_window_size
        matrix = numpy.zeros(shape=(vectorsize, len(contextWords)))
        for i in range(0, len(contextWords)):
            if i == slotposition: continue
            word = contextWords[i]
            if word == '<PAD>': continue
            if word in wordvectors:
                curvec = wordvectors[word]
            else:
                curvec = wordvectors['<UNK>']
            matrix[:,i] = curvec
        
        (matrix, meanvec) = calcInsertMeanInputs(matrix, slotposition, sum_window)
        
        newlimitedmatrix = matrix[:,slotposition-leftsize:slotposition+rightsize+1]
        newlimitedmatrix = numpy.reshape(newlimitedmatrix, vectorsize * desired_window_size)
        
        sim_vec = cosine_similarity(meanvec, typematrix)
        featurevec = numpy.concatenate((newlimitedmatrix, sim_vec[0]))
        
        myMatrix[sample, :] = featurevec
        sample += 1
    return myMatrix

def loadTypesAndVectors(targetTypesFile, vectorFile, upto=-1):
    (type2ind, n_targets, typefreq_traindev) = loadTargets(targetTypesFile)
    (wordvectors, vectorsize) = yyreadwordvectors(vectorFile, upto)
    return (type2ind, n_targets, wordvectors, vectorsize, typefreq_traindev)

def getNumberOftypeInset(resultVectorDev, tt):
    c = 0
    for t in resultVectorDev:
        if t == tt:
            c += 1
    return c

def getRandomY(resultAllVec, n, target_labels):
    new_list = list(target_labels)
    for ind in numpy.nonzero(resultAllVec)[0]:
        new_list.remove(ind)
    return random.sample(new_list, n)

def fillOnlyEntityData(myname, vectorsize, wordvectors, type2ind, n_targets, upto=-1, ds='train', binoutvec=False, convbinf=convertTargetsToBinVec):
    resultVector = []
    allTypesResultVector = []
    binntvec = []
    inputEntities = []
    tmp = getlinesm1(myname)
    numinputs = len(tmp)
    myMatrix = numpy.empty(shape=(numinputs, vectorsize))
    c = 0
    for myline in tmp:
        myline = myline.strip()
        parts = myline.split('\t')
        ent = parts[0]
        target = parts[1]
        if target not in type2ind and ds != 'test':
            print 'type not found: ' + target  
            continue;
        types_ind = []
        if len(parts) >= 3:
            other_types = parts[2].split(' ')
            for i in range(0, len(other_types)):
                if other_types[i] not in type2ind:
                    continue
                types_ind.append(type2ind[other_types[i]])    
        if ent not in wordvectors: 
            print ent + ' not in vectors'
            if ds == 'test':
                myMatrix[c, :] = numpy.zeros(shape=vectorsize, dtype=theano.config.floatX)
            else: 
                continue
        else:    
            myMatrix[c, :] = wordvectors[ent]
        inputEntities.append(ent)
        if target in type2ind:
            resultVector.append(type2ind[target])
            types_ind.append(type2ind[target])
        else:
            resultVector.append(0)
        binvec = convbinf(types_ind, n_targets)
        if binoutvec == True:
            allTypesResultVector.append(binvec)
        else:
            allTypesResultVector.append(types_ind)
        binntvec.append(convbinf([], n_targets)) #TODO: buggg    
        c += 1
        if c == upto and upto != -1:
            break
    targetMatrix = numpy.empty(shape=(c, vectorsize))
    for i in xrange(0, c):
        targetMatrix[i] = myMatrix[i] 
    print 'length targetMatrix: ' + str(len(targetMatrix))
    return(resultVector, targetMatrix, inputEntities, allTypesResultVector, binntvec)

def debug_print(var, name, PRINT_VARS=True):
    """Wrap the given Theano variable into a Print node for debugging.

    If the variable is wrapped into a Print node depends on the state of the
    PRINT_VARS variable above. If it is false, this method just returns the
    original Theano variable.
    The given variable is printed to console whenever it is used in the graph.

    Parameters
    ----------
    var : Theano variable
        variable to be wrapped
    name : str
        name of the variable in the console output

    Returns
    -------
    Theano variable
        wrapped Theano variable

    Example
    -------
    import theano.tensor as T
    d = T.dot(W, x) + b
    d = debug_print(d, 'dot_product')
    """

    if PRINT_VARS is False:
        return var

    return theano.printing.Print(name)(var)

def buildcosinematrix(matrix1, matrix2):
    """
    Calculating pairwise cosine distance using matrix1 matrix2 multiplication.
    """
    logger.info('Calculating pairwise cosine distance using matrix1 multiplication.')
    dotted = matrix1.dot(matrix2.T)
    matrix_norms = numpy.matrix(numpy.linalg.norm(matrix1, axis=1))
    matrix2_norms = numpy.matrix(numpy.linalg.norm(matrix2, axis=1))
    norms = numpy.multiply(matrix_norms.T, matrix2_norms) + 0.00000001
    sim_matrix = numpy.divide(dotted, norms)
    
    return sim_matrix


def buildtypevecmatrix(t2ind, allvectors, vectorsize, voc2idx=None):
    typevecmatrix = numpy.zeros(shape=(len(t2ind), vectorsize), dtype=numpy.float32)
    for myt in t2ind:
        if myt not in voc2idx: continue
        i = t2ind[myt]
        typevecmatrix[i] = allvectors[voc2idx[myt]]
    return typevecmatrix    
    
def extend_in_matrix(initialmatrix, newmatrix):
    """
    The two matrix should have the same row numbers. 
    """
    num_new_col = newmatrix.shape[1]
    num_old_col = initialmatrix.shape[1]
    biggermatrixtrn = numpy.zeros(shape=(len(initialmatrix), num_old_col + num_new_col))
    biggermatrixtrn[:,0:num_old_col] = initialmatrix
    biggermatrixtrn[:,num_old_col:] = newmatrix
    return biggermatrixtrn

# TODO
percentiles = [-1]
absolutes = []
Etestfreq = [1,2,5, 100]
# Etestfreq = [10]
EtestRelFreq = [10]
typefreq = [200, 3000]
ff = "{:10.3f}"
logistic = True
softmaxnorm = False

def softmax(w, t=1.0):
    """Calculate the softmax of a list of numbers w.
    @param w: list of numbers
    @return a list of the same length as w of non-negative numbers
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = numpy.exp(numpy.array(w) / t)
    dist = e / numpy.sum(e)
    return dist

def normalize(w, t=1.0):
    """Calculate the normalized: e/sum(e).
    """
    e = numpy.array(w) / t
    dist = e / numpy.sum(e)
    return dist

def divideTypes(t2f, t2i):
    t2i_list = []
    for i in range(len(typefreq) + 1):
        t2i_list.append(defaultdict(lambda: []))
    for t in t2i.keys():
        ind = 0
        for f in typefreq:
            if t2f[t][0] > f:
                ind += 1
        for i in range(ind, len(typefreq)):
            t2i_list[i][t]=t2i[t]
        if ind == len(typefreq):
            t2i_list[ind][t]=t2i[t]
    return t2i_list
    

def divideEtestByFreq(e2i, e2f, freqlist, predents, allow_miss_ent=False):
    e2i_list = []
    for i in range(len(freqlist) + 1):
        e2i_list.append(defaultdict(lambda: []))
    for e in e2i:
        if allow_miss_ent == True and e not in predents:
            continue
        ind = 0
        for f in freqlist:
            if e2f[e] > f:
                ind += 1
        for i in range(ind, len(freqlist)):
            e2i_list[i][e]=e2i[e]
        if ind == len(freqlist):
            e2i_list[ind][e]=e2i[e]
    return e2i_list
        
        

def getpercentile(mylist, mypercentile):
    if mypercentile == -1:
        return sum(mylist) / len(mylist)
    myindex = int(len(mylist) * mypercentile)
    assert myindex >= 0
    assert myindex < len(mylist)
    return mylist[myindex]

def getabsolute(mylist, myabsolute):
    assert myabsolute >= 0
    if myabsolute >= len(mylist):
        myabsolute = len(mylist) - 1
    return mylist[myabsolute]

def mynormalize(scores):
    normscores = [0.0 for i in range(len(scores))]
    mymax = max(scores)
    mymin = min(scores)
    if mymax == 0:
        return normscores
    if logistic:
        normscores = [sigmoid(scores[i]) for i in range(len(scores))] 
    elif softmaxnorm:
        normscores = softmax(scores)
    else:
        normscores = [(scores[i] - mymin) / (mymax - mymin) for i in range(len(scores))]
    return normscores
            
        
def getscores(myparts, numtype, donorm=False):
    numScorePerType = 1
    emscore = True
    if ',' in myparts[0]:
        numScorePerType = len(percentiles) + len(absolutes)
        emscore = False
    scores = [[0.0 for x in range(numtype)] for y in range(numScorePerType)]
    for i in range(numtype):
        if emscore == True:
            if i >= len(myparts):
                print myparts, numtype
            scores[0][i] = float(myparts[i])
            continue
        subparts = myparts[i].split(',')
        for ind in range(numScorePerType):
            scores[ind][i] = float(subparts[ind])
    
    if donorm and emscore == True: #emscore == True
        for ind in range(numScorePerType):
            scores[ind] = mynormalize(scores[ind])
    return (scores, numScorePerType, emscore)

def findbesttheta(unsortedlist):
    mylist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1]
    flist = []
    good = 0.0
    bad = 0.0
    for i in range(len(mylist)):
        if mylist[i][1] == 0:
            bad += 1
        elif mylist[i][1] == 1:
            good += 1
        else:
            assert 0
        prec = good / (good + bad)
        if good == 0 or total == 0:
            f = 0
        else:
            reca = good / total
            f = 2 / (1 / prec + 1 / reca)
        flist.append(f)
    mymax = max(flist)
    for i in range(len(mylist)):
        if flist[i] == mymax:
            return (mymax, mylist[i][0])
    assert 0

def computeFscore(unsortedlist, thetas):
    """
    unsortedlist is a list of tuples (score, label)
    """
    mylist = unsortedlist#sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True) 
    total = 0
    good = 0.0; bad = 0.0; fn = 0.0; tn = 0.0
    for mypair in mylist:
        total += mypair[1]
    for i in range(len(mylist)):
        if not isinstance(thetas, list):
            theta = thetas
        else:
            theta = thetas[i]
        if mylist[i][0] >= theta:
            if mylist[i][1] == 1:
                good += 1.0
            else:
                bad += 1.0
        
    return (good, bad, total);

# return number of test_lines for each testset entities
def filltest2freq(Etestfile):
    etestfile = open(Etestfile)
    e2f = {}
    for myline in etestfile:
        myparts = myline.split('\t')
        mye = myparts[0]
        if len(myparts) < 4:
            e2f[mye] = 100
        else:
            e2f[mye] = int(myparts[3])
    return e2f

def filltest2relfreq(etest2relnumfile):
    e2ffile = open(etest2relnumfile)
    e2f = {}
    for myline in e2ffile:
        myparts = myline.split(' ')
        mye = myparts[0].split('/')[2]
        e2f[mye] = int(myparts[1])
    return e2f

def readdsfile(dsfile, t2i, onlynt=False):
    e2i = {}
    etestfile = open(dsfile)
    for myline in etestfile:
        parts = myline.split('\t')
        othertypes = []
        if '/m/' not in parts[1]:
            mytype = parts[1]
        else:
            subparts = string.split(parts[1],'/')
            assert len(subparts)==3
            mytype = subparts[2]

        if len(parts) > 2:
            othertypes = parts[2].split()
        types = []
        for onet in othertypes:
            if '/m/' not in onet:
                t = onet
            else:
                t = onet.split('/')[2]
            if t in t2i and t != mytype:
                types.append(t2i[t])
        if mytype not in t2i:
            print 'mytype',mytype,'mytype'
            continue
        types.append(t2i[mytype])
        myne = parts[0].strip()
        if onlynt:
            e2i[myne] = t2i[mytype]
        else:
            e2i[myne] = types
    return e2i

def loadEnt2ScoresFile(matrix, upto, numtype, donorm=False, e2types=None):  
    print 'loading ent2type scores from', matrix  
    mtfile = open(matrix)
    lineno = -1
    t2e_scores = []
    numScorePerType = 1
    e2freq = {}
    toptrue = 0.
    for i in range(numtype):
        t2e_scores.append(defaultdict(lambda: []))
    for myline in mtfile:
        lineno += 1
        if upto>=0 and lineno>upto: break
        myparts = string.split(myline)
        myne = myparts[0]
        (scores, numScorePerType, emscore) = getscores(myparts[1:], numtype, donorm)
        if e2types and argmax(scores) in e2types[myne]:
            toptrue += 1
        for i in range(numtype):
#             if emscore == True:
#                 t2e_scores[i][myne].append(float(scores[0][i])) # this is for embedding ent2type file
#                 continue
            for j in range(numScorePerType):
                t2e_scores[i][myne].append(float(scores[j][i]))
        if len(myparts[1:]) > numtype:
            e2freq[myne] = int(myparts[numtype + 1])
    if e2types:
        print "***** p@1: ", toptrue / float(len(e2types))
        
    return (t2e_scores, numScorePerType, e2freq)

def fillEnt2scoresBaseline(e2i_test, upto, Etrainfile, t2i):
    e2tTrain = readdsfile(Etrainfile, t2i, True)
    typefreq = [0 for i in range(len(t2i))]
    big = []
    for i in range(len(t2i)):
        big.append(defaultdict(lambda: []))
    for e in e2tTrain:
#         for t in e2tTrain[e]:
        t = e2tTrain[e]
        typefreq[t] += 1
    for mye in e2i_test:
        for i in range(len(t2i)):
            big[i][mye].append(typefreq[i])
    return big 
    
def fillt2i(typefilename):
    t2i = {}
    t2f = {}
    typefile = open(typefilename)
    i = -1
    for myline in typefile:
        i += 1
        if '/m/' in myline:
            myparts = string.split(myline)
            subparts = string.split(myparts[0],'/')
            assert len(subparts)==3
            assert subparts[1]=='m'
            t2i[subparts[2]] = i
        else:
            myparts = myline.split('\t')
            assert len(myparts) == 3
            t2i[myparts[0]] = i
            etrn_freq = int(myparts[1])
            contextfreq = int(myparts[2])
            t2f[myparts[0]] = (etrn_freq, contextfreq)
    return (t2i,t2f)
def calcPRF(good, bad, total):
    if (good + bad) == 0.0:
        prec = 0
    else:
        prec = good / (good + bad)
    if good == 0.0 or total == 0.0:
        f = 0.0
        reca = 0.0
    else:
        reca = good / total
        f = 2.0 / (1 / prec + 1 / reca)
    return (prec, reca, f);

def calcNNLBmeasures(unsortedlist, mintopscore=0.0):
    sortedlist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    goodAt1 = sortedlist[0][1]
    topscore = sortedlist[0][0]
    best = findbesttheta(sortedlist)
    return (goodAt1, best[0], topscore)

def calcMeasuresBaseline(bigBaseline, e2i, numtype, onlynt='False'):
    goodsbase= 0.0; fbase = 0.0
    for mye in e2i.keys():
        e2tscores = []
        etypes = e2i[mye]
        for i in range(numtype):
                correct = 0
                if onlynt == 'False' and i in etypes:
                    correct = 1 
                elif onlynt == 'True' and i == etypes:
                    correct = 1
                e2tscores.append((bigBaseline[i][mye], correct))
        (goodAt1, fnn, topscore) = calcNNLBmeasures(e2tscores)
        goodsbase += goodAt1; 
        fbase += fnn
    prec1 = goodsbase / len(e2i)
    f = fbase / len(e2i)    
    return (prec1, f)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def precisionAt(unsortedlist, topnum=20):
    mylist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1] #mypair[1] is one for labeled entity and type
    if total < topnum:
        topnum = total
    good = 0.0; bad = 0.0; fn = 0.0; tn = 0.0

    for i in range(0, topnum):
        if mylist[i][1] == 1:
            good += 1
    if topnum == 0:
        return 0.0
    return good / topnum

def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return list_of_ele[0]

padTag = '<PAD>'
unkTag = '<UNK>'
startTag = '<S>'
endTag = '</S>'

def build_char_vocab(ds2names):
    MINIMUM_FREQ = len(ds2names) / 1000
    char2freq = defaultdict(lambda: 0)
    for names in ds2names.values():
        for onename in names:
            for c in onename:
                char2freq[c] += 1
    char_vocab = []
    for c in char2freq:
        if char2freq[c] < MINIMUM_FREQ:
            continue
        char_vocab.append(c)
    char_vocab.extend([padTag, unkTag, startTag, endTag])
    return char_vocab
        

def get_ngram_seq(voc2idx, seq, max_len=100):
    ngramseq = [voc2idx[startTag]]
    for i in range(0, max_len - 1):
        if i == len(seq):
            ch = endTag
        elif i > len(seq):
            ch = padTag
        else:
            ch = seq[i]
            if ch not in voc2idx:
                ch = unkTag
        ngramseq.append(voc2idx[ch])
    assert len(ngramseq) == max_len
    return numpy.array(ngramseq)

def build_ngram_vocab(letter_matrix, ngram):
    logger.info('building vocab for ngram of characters')
    MIN_FREQ = 50
    ngram2idx= {}; idx2ngram = {}
    idx = 0
    ng2freq = defaultdict(lambda: 0)
    for inst in letter_matrix:
        for i in range(len(inst) - ngram + 1):
            ng = ' '.join(str(l) for l in inst[i:i + ngram])
            ng2freq[ng] += 1
    print 'original size of ngrams', len(ng2freq)
    for inst in letter_matrix:
        for i in range(len(inst) - ngram + 1):
            ng = ' '.join(str(l) for l in inst[i:i + ngram])
            if ng2freq[ng] < MIN_FREQ:
                continue
            if ng not in ngram2idx:
                ngram2idx[ng] = idx
                idx2ngram[idx] = ngram
                idx += 1
    
    for i, tag in enumerate([padTag, unkTag, startTag, endTag]):
        ngram2idx[tag] = idx
        idx2ngram[idx] = tag
        idx += 1
    logger.info('vocab size for ngram: %d is %d', ngram, len(ngram2idx))
    return ngram2idx, idx2ngram

# def get_ngram_seq(let_seq, ngram2idx, ngram):
#     ng_seq = [ngram2idx[unkTag]]
#     for i in range(len(let_seq) - ngram + 1):
#         ng = ' '.join(str(l) for l in let_seq[i:i + ngram])
#         if ng not in ngram2idx:
#             logger.error('ngram: %s not in the ngramvocab', ng)
#             ng = unkTag
#         ng_seq.append(ngram2idx[ng])
#     ng_seq.append(ngram2idx[endTag])
#     return ng_seq

def filtertypes(enttypes, t2idx, use_ix=False, force_notable=True):
    if force_notable and enttypes[0] not in t2idx: 
        return []
    outtypes = []
    for myt in enttypes:
        if myt in t2idx:
            if use_ix == True:
                outtypes.append(t2idx[myt])
            else:
                outtypes.append(myt)
    return outtypes


def refine_names(names):
    filnames = []
    for n in names:
        if n.strip() == '' or n.strip() == ' ':
            continue
        filnames.append(n)
    return filnames

def get_ent_names(names, maxnum=3):
    newnames = refine_names(names) #remove empy names
    if len(names) >= maxnum:
        return names[:maxnum];
    if len(names) == 0:
        thename = ''
    else:
        thename = names[0]
    for _ in range(maxnum - len(names)):
        newnames.append(thename)
    return newnames

def load_entname_ds(dsfile, t2idx, use_ix=False):
    logger.info('loading dataset %s', dsfile)
    f = open(dsfile)
    e2types = defaultdict()
    e2names = OrderedDict()
    e2freq = defaultdict(lambda: 0)
    for line in f:
        parts = line.strip().split('\t')
        types = [parts[1].strip()]
        if len(parts) > 2:
            for t in parts[2].split():
                if t not in types:
                    types.append(t)
        filtered_types = filtertypes(types, t2idx, use_ix, force_notable=False)
        if len(filtered_types) == 0:
            continue
        e2types[parts[0]] = filtered_types
        if len(parts) < 4:
            print parts
            continue
        e2freq[parts[0]] = int(parts[3])
        thenames = []
        for i in range(5,len(parts), 2):
            thenames.append(parts[i])
        e2names[parts[0]] = thenames

    return (e2types, e2names, e2freq) 

def loadtypes(filename):
    t2ind = dict()
    ind2t = dict()
    c = 0
    f = open(filename, 'r')
    for line in f:
        t = line.split('\t')[0].strip()
        t2ind[t] = c
        ind2t[c] = t
        c += 1
    return (t2ind, ind2t)

def read_embeddings_vocab(fname, vocab=None, num=-1):
    """ Read word embeddings from file

    :param embedding_file:  Path to embedding file
    :type embedding_file:   str/unicode
    :param upto:             Restrict number of embeddings to load
    :type upto:              int
    :returns:               Mapping from words to their embedding
    :rtype:                 str -> numpy.array dict
    """
    # NOTE: This should be pretty efficient, since we're only reading the file
    #       line by line and are reading directly into memory-efficient numpy
    #       arrays.
    with open(fname) as fp:
        _, vec_size = (int(x) for x in fp.readline().strip().split())
        #embeddings = numpy.zeros((len(vocab), vec_size), dtype='float32')
        embeddings = numpy.random.random_sample((len(vocab), vec_size))
#       num_vecs += 1  # For <UNK>
        nwwemb = 0 #number of vocab words with embeddings
        for idx, line in enumerate(fp, start=1):
            if num != -1 and idx > num:
                break
            parts = line.strip().split()
            if vocab is not None and parts[0] not in vocab:
                continue
            nwwemb += 1
            embeddings[vocab[parts[0]]] = [float(v) for v in parts[1:]]
        embeddings[vocab[unkTag]] = numpy.mean(embeddings, axis=0)
        embeddings[vocab[padTag]] = numpy.zeros((vec_size,))
    #rare words are unknown and words that do not have embeddings have zero
    logger.info('number of vocabs that have word2vec embeddings: %d', nwwemb)  
    return embeddings, vec_size

def build_type2entmatrix(bigmatrix, fblist, t2i, norm=False):
    lineno = -1
    big = []
    for i in range(len(t2i)):
        big.append(collections.defaultdict(lambda: []))
    for i in xrange(len(bigmatrix)):
#         print str(lineno)
        lineno += 1
        scores = bigmatrix[i]
        myne = fblist[lineno][0]
        if norm:
            scores = mynormalize(scores)
        for i in range(len(t2i)):
            big[i][myne].append(scores[i])
    return big

def big2small(big, t2i):
    small = []
    for i in range(len(t2i)):
        small.append(collections.defaultdict(lambda: []))
            
    for i in range(len(t2i)):
        for mye in big[i]:
            big[i][mye].sort()
            big[i][mye].reverse()
            for mypercentile in percentiles:
                summaryScore = getpercentile(big[i][mye],mypercentile)
                small[i][mye].append(summaryScore)
#             for abos in absolutes:
#                 summaryScore = getabsolute(big[i][mye], abos)
#                 small[i][mye].append(summaryScore)
            for myabsolute in absolutes:
                summaryscore = getabsolute(big[i][mye],myabsolute)
                small[i][mye].append(summaryscore)
    return small
    
def big2small_avgs(big, t2i, top_perc=0.2):
    small = []
    for i in range(len(t2i)):
        small.append(collections.defaultdict(lambda: []))
    for i in range(len(t2i)):
        for mye in big[i]:
            big[i][mye].sort()
#             big[i][mye].reverse()
            my_idx = int((1. - top_perc) * len(big[i][mye]))
            summaryScore = sum(big[i][mye][my_idx:]) / len(big[i][mye][my_idx:])
            small[i][mye].append(summaryScore)
    return small


def write_small(myfile, small, t2i, e2freq):
    with open(myfile, 'w') as fp:
        for mye in small[0]:
            line = '/m/' + mye 
            for i in range(len(t2i)):
                line += ' '
                line += ','.join([str(sc) for sc in small[i][mye]])
            line += ' ' + str(e2freq[mye])
            fp.write(line + '\n')
    logger.info('small matrix saved in: %s', myfile)

def write_small_multi(myfile, small, e2freq, ent_list, num_labels=102):
    with open(myfile, 'w') as fp:
        for ent_idx, mye in enumerate(ent_list):
            line = '/m/' + mye + ' ' 
            line += ' '.join([str(sc) for sc in small[ent_idx]])
            line += ' ' + str(e2freq[mye])
            fp.write(line + '\n')
    logger.info('small matrix saved in: %s', myfile)
    
def load_lines_info(fbfile, t2i, top=None):
    fblist = []
    lineno = -1
    e2freq = defaultdict(lambda: 0)
    ent_list = OrderedDict()
    line_list = []
    with open(fbfile) as fp:
        for myline in fp:
            lineno += 1
            if top and lineno >= top:
                break 
            myparts = myline.split('\t')
            mytype = myparts[2]
            if '/m/' in myparts[2]:
                subparts = string.split(myparts[2],'(')
                assert len(subparts)==2
                subsubparts = string.split(subparts[0],'/')
                assert len(subsubparts)==3
                mytype = subsubparts[2]
            if mytype not in t2i:
                logger.error('subsub',subsubparts)
                assert 0
            myne = myparts[1].split('/')[2]
            fblist.append((myne,t2i[mytype]))
            e2freq[myne] += 1
            all_types = [mytype]
            all_types.extend(myparts[3].split(','))
            ent_list[myne] = e2freq[myne]
            line_list.append((all_types, myparts[4].strip()))
            
    logger.info('..loaded with size: %d', len(fblist))
    return fblist, e2freq, ent_list, line_list

def loadTargets(filename):
    # Filling type indexes from a type list file
    targetIndMap = dict()
    typefreq_traindev = dict()
    c = 0
    f = open(filename, 'r')
    for line in f:
        t = line.split('\t')[0].strip()
        targetIndMap[t] = c
        if len(line.split('\t')) < 3:
            typefreq = 0
        else:
            typefreq = line.split('\t')[2].strip()
        if typefreq == 'null':
            typefreq_traindev[t] = 0
        else:
            typefreq_traindev[t] = int(typefreq)
        c += 1
    return (targetIndMap, len(targetIndMap), typefreq_traindev) 

def getsentences(text):
    #sents = nltk.sent_tokenize(text)
    return re.compile(" [\?\!\.]").split(text)
    #return text.split(' .')

def parseents(mystr):
    ents = mystr.split(' ')
    mids = []
    for e in ents:
        mids.append('/m/' + e.split('/')[2])
    return mids 

def has_ent(sent, ent_mid):
#     for mid in ent_mids:
    if ent_mid in sent:
        return True
    return False
def getentparts(ent_token):
    ent_parts = ent_token.split('##')
    parts = ent_parts[0].split('/')
    if len(parts) < 4:
        return (ent_parts[0], 'unk', ent_parts[1])
    mid = '/m/' + parts[2]
    tokens = parts[3].split('_')
    if len(ent_parts) < 2:
        print ent_token
    notabletype = ent_parts[1]
    return (mid, tokens, notabletype)

def getentparts2(ent_token):
    ent_parts = ent_token.split('##')
    parts = ent_parts[0].split('/')
    if len(parts) < 4:
        return (ent_parts[0], 'unk', 1)
    mid = '/m/' + parts[2]
    name = parts[3]
    if len(ent_parts) < 2:
        print ent_token
    notabletype = ent_parts[1]
    return (mid, name, notabletype)


def readall(filename):
    myfile = open(filename, 'r')
    mystring = myfile.read()
    myfile.close()
    return mystring

def getfilelines(filename, upto=-1):
    fcontents = readall(filename)
    lines = string.split(fcontents, '\n')
    if upto > 0:
        lines = lines[:upto]
    if len(string.strip(lines[-1])) == 0:
        lines = lines[:-1]
    return lines

def load_dataset(dsfile, logger=logger):
    logger.info('loading dataset %s', dsfile)
    f = open(dsfile)
    e2types = defaultdict()
    t2ents = defaultdict(set)
    e2freq = defaultdict(lambda : 0)
    for line in f:
        parts = line.split('\t')
        types = [parts[1].strip()]
        for t in parts[2].split():
            types.append(t)
        e2types[parts[0]] = types
        t2ents[parts[1].strip()].add(parts[0])
        if len(parts) > 3:
            e2freq[parts[0]] = int(parts[3])
    return (e2types, t2ents, e2freq)

def filltest2freq_(Etestfile):
    etestfile = open(Etestfile)
    e2f = {}
    for myline in etestfile:
        myparts = myline.split('\t')
        mye = myparts[0]
        if len(myparts) < 4:
            e2f[mye] = 100
        else:
            e2f[mye] = int(myparts[3])
    return e2f


def writelines(thelines, outfile):
    f = open(outfile, 'w')
    for l in thelines:
        f.write(l.strip() + '\n')
    f.close()
        
    
def write_ds(myent2types, e2freq, outfile):
    f = open(outfile, 'w')
    for mye in myent2types:
        f.write(mye + '\t' + myent2types[mye][0] + '\t' + ' '.join(myent2types[mye][1:]) + '\t' + str(e2freq[mye]))
        f.write('\n')
    f.close()
