# for debugging: upto = 10000 will only look at the first 10000 lines

#finding thresholds from summarized scores for each entity2type

from myutils import * 
import string,collections, sys
import yaml
from numpy import mean

# Thresholds are found and put into typethresholdMatrix--- Now we should apply thresholds and calc performance
def calcPrintMeasures(myetest, myType2threshMatrix, smalltest, findGoodEnts=False):
    ######
    if len(myetest) == 0:
        print 'no test set with these conditions'
        return 
    correctEntititesP1 = []
    
    entityFscoreMatrix = [[0 for x in range(numScorePerType)] for x in range(len(myetest))] 
    fLooseMacro = [0.0 for x in range(numScorePerType)]
    precLooseMacro = [0.0 for x in range(numScorePerType)]
    recLooseMacro = [0.0 for x in range(numScorePerType)]
    goods = [0.0 for i in range(numScorePerType)]; bads = [0.0 for ind in range(numScorePerType)]; totals = [0.0 for ind in range(numScorePerType)];
    goodsAt1 = [0.0 for i in range(numScorePerType)]; f_nnlb = [0.0 for ind in range(numScorePerType)]; 
    strictGoods = [0.0 for i in range(numScorePerType)]
    
    for mye in myetest:
        e2tscores = []
        etypes = myetest[mye]
        for j in range(numScorePerType):
            e2tscores.append([])
            for i in range(numtype):
                correct = 0
                if onlynt == False and i in etypes:
                    correct = 1
                elif onlynt == True and i == etypes:
                    correct = 1
                if mye not in smalltest[i]:
                    e2tscores[j].append((0.0, correct))
                else:
                    e2tscores[j].append((smalltest[i][mye][j], correct))
        ind = -1
        for sublist in e2tscores:
            ind += 1
            (good, bad, total) = computeFscore(sublist, [myType2threshMatrix[i][ind] for i in range(numtype)])
#             (good, bad, total) = computeFscore(sublist, [0.8 for i in range(numtype)])
    #         print mye, 'good: ', good, 'bad: ', bad, 'total: ', total
            (p,r,f) = calcPRF(good, bad, total)
            
            goods[ind] += good; bads[ind] += bad; totals[ind] += total
            precLooseMacro[ind] += p; 
            recLooseMacro[ind] += r
            fLooseMacro[ind] += f                             
            if good == total and bad == 0: 
                strictGoods[ind] += 1
            (goodAt1, fnn, topscore) = calcNNLBmeasures(sublist)
            goodsAt1[ind] += goodAt1; f_nnlb[ind] += fnn
            
            if goodAt1 > 1:
                correctEntititesP1.append(mye)
    
    
    print '**Prec strict (per entity)---'
    for i in range(numScorePerType):
        print i, ff.format(strictGoods[i] / len(myetest))
    
    print '**Macro (per entity) --- '
    for i in range(numScorePerType):
        recLooseMacro[i] /= len(myetest)
        precLooseMacro[i] /= len(myetest)
        fLooseMacro[i] /= len(myetest)
        print i, " Prec: ", ff.format(precLooseMacro[i]), ' Reca: ', ff.format(recLooseMacro[i]), ' F1: ', ff.format(fLooseMacro[i])  
    
    print '**Micro (per entity or per type) --- '
    for i in range(numScorePerType):
        (pr , re, f ) = calcPRF(goods[i], bads[i], totals[i])
        print i, 'Prec: ', ff.format(pr), ' Reca: ', ff.format(re), ' F1: ',ff.format(f) 
    
    print '**measures based on NNLB: '
    for i in range(numScorePerType):
        print i, 'Prec@1: ', ff.format(goodsAt1[i] / len(myetest)), 'Avg F1: ', ff.format(f_nnlb[i] / len(myetest))
        
def calcPrintBaseline(smallTstMFT, mye2i, numtype, onlynt):
    if len(mye2i) == 0:
        return
    print len(mye2i)
    (prec1, f) = calcMeasuresBaseline(smallTstMFT, mye2i, numtype, onlynt)
    print '**Most Frequent types baseline (based on NNLB): '
    print 'Prec@1: ', ff.format(prec1), 'Avg F1: ', ff.format(f)





def fillthresholmatrix(edev2freq, smalldev, e2i_dev, numScorePerType):
    typethresholdMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)]
    firstrun = True
    for i in range(numtype):
        thelist = []
        for j in range(numScorePerType):
            thelist.append([])
            for mye in e2i_dev:
                if mye in edev2freq and edev2freq[mye] < 0:
                    continue
                if mye not in smalldev[i]: continue
                correct = 0
                if onlynt == False and i in e2i_dev[mye]:
                    correct = 1
                elif onlynt == True and i == e2i_dev[mye]:
                    correct = 1
                thelist[j].append((smalldev[i][mye][j], correct))
        ind = -1
        for sublist in thelist:
            ind += 1
            best = findbesttheta(sublist)
            typethresholdMatrix[i][ind] = best[1]
    return typethresholdMatrix

def calc_AP(scores_list):
    mylist = sorted(scores_list, key=lambda tuple: tuple[0], reverse=True)
    rel_docs_up_to_i = 0.0
    prec_list = []
    for i in range(len(mylist)):
        if mylist[i][1] == 1: # found a relevant document
            rel_docs_up_to_i += 1.
            prec = rel_docs_up_to_i / (i + 1)
            prec_list.append(prec)
    if len(prec_list) == 0:
        return 0.
    return mean(numpy.asarray(prec_list))

def calc_average_precision(small, e2i, numScorePerType=1, e2freq={}, numtype=102, onlynt=False):
    type_to_AP = defaultdict(list)
    for i in range(numtype):
        thelist = []
        for j in range(numScorePerType):
            thelist.append([])
            for mye in e2i:
                if mye in e2freq and e2freq[mye] < 0:
                    continue
                if mye not in small[i]: continue
                correct = 0
                if onlynt == False and i in e2i[mye]:
                    correct = 1
                elif onlynt == True and i == e2i[mye]:
                    correct = 1
                thelist[j].append((small[i][mye][j], correct))
        ind = -1
        for sublist in thelist:
            ind += 1
            type_to_AP[ind].append(calc_AP(sublist))
    return type_to_AP

def calc_MAP(type_to_AP):
    for ind in type_to_AP:
        mean = numpy.mean(numpy.asarray(type_to_AP[ind]))
        print 'MAP: ', mean
            

def print_freq_details():
    print allow_missing_entity
    e2i_test_list = divideEtestByFreq(e2i_test, etest2f, Etestfreq, smalltest[0], allow_missing_entity)
    assert len(e2i_test_list) == len(Etestfreq) + 1
    
    for i in range(len(Etestfreq) - 1):
        print '-------\nresult for Etest with freq <=',Etestfreq[i]
        myetest = e2i_test_list[i]
        print 'num of entities:', len(myetest)
        calcPrintMeasures(myetest, typethresholdMatrix,smalltest)

    i = len(Etestfreq)
    print '-------\nresult for Etest with freq >', Etestfreq[i - 1]
    myetest = e2i_test_list[i]
    print 'num of entities:', len(myetest)
    if len(myetest) > 0:
        calcPrintMeasures(myetest, typethresholdMatrix,smalltest)
    
    print '-------\nresult for All Etest entities'
    print 'num of entities:', len(e2i_test)
    calcPrintMeasures(e2i_test, typethresholdMatrix, smalltest, findGoodEnts=True)


def fill_name_vocab(e2names_train, maxname=3):
    vocab = set()
    for names in e2names_train.values():
        for name in names[:maxname]:
            for tok in name.split():
                vocab.add(tok)
    return vocab


def divide_unknown(e2names, e2types, vocab, maxname=1):
    knwnents = defaultdict(list); unkents = defaultdict(list)
    for mye in e2names:
        names = e2names[mye][:maxname]
        found = False
        for onename in names:
            for tok in onename.split():
                if tok in vocab:
                    knwnents[mye] = e2types[mye]
                    found = True
        if found == False: unkents[mye] = e2types[mye]
    return knwnents, unkents


def print_res(mye2t_test, mye2t_dev):
#     thresholdmatrix = fillthresholmatrix(edev2freq, smalldev, mye2t_dev, numScorePerType)
#     thresholdmatrix = fillthresholmatrix(edev2freq, smalldev, e2i_dev, numScorePerType)
    calcPrintMeasures(mye2t_test, typethresholdMatrix, smalltest)
    

def print_known_ukn_details(max_names=[3,1,1]):
    vocab = fill_name_vocab(e2names_train, maxname=max_names[0])
    print 'vocabsize=', len(vocab)
    known_dev, ukn_dev = divide_unknown(e2names_dev, e2i_dev, vocab, maxname=max_names[1])
    known_tst, ukn_tst = divide_unknown(e2names_test, e2i_test, vocab, maxname=max_names[2])
    print len(known_tst), len(ukn_tst), len(known_dev), len(ukn_dev)
    print '\nResults for known entities: entities with one token in the training vocab'
    print_res(known_tst, known_dev)
    print '\nResults for UNknown entities: entities with NO token in the training vocab'
    print_res(ukn_tst, ukn_dev)
    

def print_MFT_baseline():
    e2i_test_list = divideEtestByFreq(e2i_test, etest2f, Etestfreq, smalltest[0], allow_missing_entity)
    assert len(e2i_test_list) == len(Etestfreq) + 1
    
    for i in range(len(Etestfreq) - 1):
        print '-------\nresult for Etest with freq <=',Etestfreq[i]
        myetest = e2i_test_list[i]
        print 'num of entities:', len(myetest)
        calcPrintMeasures(myetest, t2thrMFT,smallTstMFT)

    i = len(Etestfreq)
    print '-------\nresult for Etest with freq >', Etestfreq[i - 1]
    myetest = e2i_test_list[i]
    print 'num of entities:', len(myetest)
    if len(myetest) > 0:
        calcPrintMeasures(myetest, t2thrMFT,smallTstMFT)
    
    print '-------\nresult for All Etest entities'
    print 'num of entities:', len(e2i_test)
    calcPrintMeasures(e2i_test, t2thrMFT,smallTstMFT)

if __name__ == '__main__':
    upto = -1
    cnfpath = sys.argv[1]
    if '.yaml' in cnfpath: 
        with open(cnfpath) as fp:
            config = yaml.load(fp)
    else:
        config = loadConfig(cnfpath)
        
    print 'loading cofing ',config
    numtype = int(config['numtype']) if 'numtype' in config else 102
    Etestfile = config['Etest']
    Edevfile = config['Edev'] 
    Etrainfile = config['Etrain']
    matrixdev = config['aggmatrixdev']  if 'aggmatrixdev' in config else config['matrixdev'] 
    matrixtest = config['aggmatrixtest']  if 'aggmatrixtest' in config else config['matrixtest'] 
    typefilename = config['typefile']
    freqtype = 'mention'
    if 'ent_freq_type' in config:
        freqtype = config['ent_freq_type']
    donorm = str_to_bool(config['norm']) 
    onlynt = str_to_bool(config['onlynt']) if 'onlynt' in config else False
    print '** norm is ', donorm
    print '* onlynt is', onlynt
    allow_missing_entity = True
    if 'allow_miss_ent' in config:
        allow_missing_entity = config['allow_miss_ent']
    print '***', allow_missing_entity
    (t2ix,t2f) = fillt2i(typefilename)
    e2i_train, e2names_train,_ = load_entname_ds(Etrainfile, t2ix, True)
    e2i_dev, e2names_dev,_ = load_entname_ds(Edevfile, t2ix, True)
    e2i_test, e2names_test,_ = load_entname_ds(Etestfile, t2ix, True)
    (smalldev, numScorePerType, edev2freq) = loadEnt2ScoresFile(matrixdev, upto, numtype, donorm, e2i_dev)
    (smalltest, numScorePerTypetest, etest2freq) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm, e2i_test)
    print 'num of entities in smalltest: ', len(smalltest[0]), 'and number of test entities: ', len(e2i_test)
    print 'num of entities in smalldev: ', len(smalldev[0]), 'and number of dev entities: ', len(e2i_dev)
    etest2f = filltest2freq(Etestfile)
    if False:
        smallTstMFT = fillEnt2scoresBaseline(e2i_test, upto, Etrainfile, t2ix)
        smallDevMFT = fillEnt2scoresBaseline(e2i_dev, upto, Etrainfile, t2ix)
        t2thrMFT = fillthresholmatrix(edev2freq, smallDevMFT, e2i_dev, numScorePerType)
        print t2thrMFT
        #calcPrintMeasures(e2i_test, t2thrMFT, smallTstMFT)
        type_to_AP = calc_average_precision(smallTstMFT, e2i_test, numScorePerType=1, e2freq=etest2freq)
        calc_MAP(type_to_AP)
        print_MFT_baseline()
    
    type_to_AP = calc_average_precision(smalltest, e2i_test, numScorePerType=1)
    calc_MAP(type_to_AP)
    
    typethresholdMatrix = fillthresholmatrix(edev2freq, smalldev, e2i_dev, numScorePerType)
    
    
    print_freq_details()
    print '#####################'
#     sys.exit()
    if 'name' in Etrainfile:
        max_names = [int(n) for n in config['name_num'].split()] if 'name_num' in config else [3,1,1] # index 0 for train, 1 for dev, 2 for test
        print_known_ukn_details(max_names)
    
    
        
        
    
    
    
