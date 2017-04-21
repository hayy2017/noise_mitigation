'''
Created on Feb 21, 2016

@author: yy1
'''
from myclasses import Mention
from _collections import defaultdict
from myutils import loadConfig
import sys
def load_contexts(path, upto=-1):
    mid2contexts = defaultdict(list)
    with open(path) as fp:
        for c, line in enumerate(fp):
            line = line.strip()
            if upto != -1 and c > upto:
                break
            #print line, len(line.split('\t'))
            if len(line.split('\t')) > 6:
                m = Mention.parse_line_new(line)
            else:
                m = Mention.parse_line_old(line)
            if m == None: continue
            mid2contexts[m.fb_mid].append(m)
    print len(mid2contexts), 'entities are loaded from: ', path
    return mid2contexts

import numpy, numpy.random
def write_extended_mentions(path, e2mentions, count=300):
    fp = open(path, 'w')
    for mye in e2mentions:
        mentions = e2mentions[mye]
        if len(mentions) > count:
            selmentions = list(numpy.random.choice(mentions, size=count, replace=False))
        else:
            selmentions = mentions + list(numpy.random.choice(mentions, size=count-len(mentions)))
        for mymnt in selmentions:
            theline = mymnt.get_line_my_oldformat()

            fp.write(theline + '\n')
    fp.close()

def write_mentions(path, e2mentions):
    fp = open(path, 'w')
    for mye in e2mentions:
        mentions = e2mentions[mye]
        for mymnt in mentions:
            theline = mymnt.get_line_my_oldformat_typeInContext()
            fp.write(theline + '\n')
    fp.close()
    
if __name__ == '__main__':
    config = loadConfig(sys.argv[1])
    dev = config['testcontexts']
    edev2mentions = load_contexts(dev, -1)
#     write_extended_mentions(dev+'.multi', edev2mentions, count=300)
    write_mentions(dev+'.ordered', edev2mentions)
    sys.exit()
    dev = config['devcontexts']
    edev2mentions = load_contexts(dev, -1)
#     write_extended_mentions(dev+'.multi', edev2mentions, count=300)
    write_mentions(dev+'.ordered', edev2mentions)
    
    dev = config['devcontexts_big']
    edev2mentions = load_contexts(dev, -1)
#     write_extended_mentions(dev+'.multi', edev2mentions, count=100) #count would be our batch size, each batch for one entity
    write_mentions(dev+'.ordered', edev2mentions)
    dev = config['traincontexts']
    edev2mentions = load_contexts(dev, -1)
#     write_extended_mentions(dev+'.multi', edev2mentions, count=100)
    write_mentions(dev+'.ordered', edev2mentions)
    
    
