'''
Created on Feb 9, 2016

@author: yadollah
'''
from _collections import defaultdict
import operator

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


def get_types_myformat(types):
        return [t.replace('/', '-') for t in types]
    
class Mention(object):
    def __init__(self, idx, types, entityname=None, start=None, end=None, tokens=None, mycontext=None, fb_mid=None):
        self.idx = idx
        self.types = get_types_myformat(types)
        self.fb_mid = fb_mid
        self.entityname = entityname
        if mycontext is None:
            self.start = start
            self.end = end
            self.tokens = tokens
        else:
            self.mycontext = mycontext
            self.start = None
            self.end = None
            self.get_tokens_from_mycontext()
        self.mentionName = '_'.join(self.tokens[self.start:self.end])

    def get_line_my_format(self):
        mycontext = []
        othtypes = ','.join(self.types[1:])
        for j, _ in enumerate(self.tokens):
            if j == self.start:
                mycontext.append(self.fb_mid + '/' + '_'.join(self.tokens[self.start:self.end]) + '##' + self.types[0])
            elif j > self.start and j < self.end:
                continue
            else:
                mycontext.append(self.tokens[j])
        self.mycontext = mycontext
        self.myline = '\t'.join([str(self.idx), str(self.start) + ','+ str(self.end), self.entityname, self.fb_mid, self.types[0], othtypes, ' '.join(mycontext)])
        return self.myline
   
    def get_line_my_oldformat(self):
        mycontext = []
        othtypes = ','.join(self.types[1:])
        for j, _ in enumerate(self.tokens):
            if j == self.start:
                mycontext.append(self.fb_mid + '/' + '_'.join(self.tokens[self.start:self.end]) + '##' + self.types[0])
            elif j > self.start and j < self.end:
                continue
            else:
                mycontext.append(self.tokens[j])
        self.mycontext = mycontext
        self.myline = '\t'.join([str(self.idx), self.fb_mid, self.types[0], othtypes, ' '.join(mycontext)])
        return self.myline

    def get_line_my_oldformat_typeInContext(self):
        othtypes = ','.join(self.types[1:])
        self.myline = '\t'.join([str(self.idx), self.fb_mid, self.types[0], othtypes, ' '.join(self.mycontext)])
        return self.myline
    

    def get_tokens_from_mycontext(self):
        tokens = []
        i = 0
        for t in self.mycontext:
            if '/m/' in t:
                mid, ttt, _ = getentparts(t)
                tokens.extend(ttt)
                if mid == self.fb_mid:
                    self.start = i
                    self.end = self.start + len(ttt)
                i += len(ttt)
            else:
                tokens.append(t)
                i += 1
        self.tokens = tokens
        return tokens    
    
    @staticmethod
    def parse_line_new(myline):
        (idx, stend, wikiName, mid, nottype, othertypes, context) = myline.strip().split('\t')
        types = [nottype]
        for myt in othertypes.split(','):
            if len(myt) > 1:
                types.append(myt)
        return Mention(idx, types, entityname=wikiName, mycontext=context, fb_mid=mid)

    @staticmethod
    def parse_line_old(myline):
        (idx, mid, nottype, othertypes, context) = myline.strip().split('\t')
        types = [nottype]
        if ',' in othertypes:
            for myt in othertypes.split(','):
                if len(myt) > 1:
                    types.append(myt)
        else:
            for myt in othertypes.split(' '):
                if len(myt) > 1:
                    types.append(myt)
        mycontext = context.split(' ')
        if mid not in context:
            return None
        return Mention(idx, types, mycontext=mycontext, fb_mid=mid)

class KbEntity(object):
    def __init__(self, mid, types, wiki_name=None, aliases=None, nottype=None):
        self.mid = mid
        self.types = types
        self.nottype = nottype if nottype is not None else types[0]
        self.wikiname = wiki_name
        self.aliases = aliases
        
    def __hash__(self):
        return hash(self.mid)

    def __eq__(self, other):
        return (self.mid) == (other.mid)
    
    def __str__(self):
        return self.mid
        
        
class EntityDataset(object):
    def __init__(self, ent2mentions=None, ent2name2freq=None, ent2freq=None):
        """
        :param ent2mentions: dict with KbEntity to its mentions with type Mention
        """
        self.ent2mentions = ent2mentions
        self.ent2name2freq = self.get_ent2aliases() if ent2mentions is not None else ent2name2freq
        self.ent2freq = ent2freq if ent2freq is not None else {key: len(value) for key, value in self.ent2mentions.items()} 
    
    def get_ent2aliases(self):
        ent2name2freq = defaultdict(lambda: defaultdict(lambda:0))
        for ent in self.ent2mentions:
            for m in self.ent2mentions[ent]:
                ent2name2freq[ent][m.mentionName] += 1
        return ent2name2freq
        
    def write_to_file(self, stream):
        """
        writing entity dataset to the file
        """
        for ent in self.ent2mentions:
            names_str = ''
            name2freq = self.ent2name2freq[ent]
            sorted_by_freq = sorted(name2freq.items(), key=operator.itemgetter(1), reverse=True)
            for name, freq in sorted_by_freq:
                names_str += ' '.join([name, str(freq)]) + '\t'
            myline = '\t'.join([ent.mid, ent.nottype, ' '.join(ent.types), str(len(self.ent2mentions[ent])), '####', names_str])
            stream.write(myline + '\n')
    
    @staticmethod 
    def parse_one_line(myline):
        myparts = myline.strip().split('\t')
        assert len(myparts) > 5
        mid, nottype, types, freq, _ = myparts[:5]
        typelist = types.strip().split(' ')
        name2freq = {}
        for nf in myparts[5:]:
            name2freq[nf.split(' ')[0]] = int(nf.split(' ')[1])
        return KbEntity(mid, typelist, nottype=nottype), name2freq, int(freq)
            
    @staticmethod
    def load_ents_from_file(entityfile):
        print 'loading entity daset from file: ', entityfile
        ent2name2freq = defaultdict(lambda: defaultdict(lambda:0))
        ent2freq = {} 
        with open(entityfile) as fp:
            for myline in fp:
                entobj, name2freq, entfreq = EntityDataset.parse_one_line(myline)
                ent2name2freq[entobj] = name2freq
                ent2freq[entobj] = entfreq
        return EntityDataset(ent2mentions=None, ent2name2freq=ent2name2freq, ent2freq=ent2freq)
        
    
