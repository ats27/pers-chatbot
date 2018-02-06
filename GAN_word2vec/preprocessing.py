import math
import re
import unicodedata
import time
from torch.autograd import Variable
import torch
from io import open
import numpy as np
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Config():
    def __init__(self,use_cuda):
        self.use_cuda = use_cuda

    def load_parameters(self,model,name):
        if name=="encoder":
            model.load_state_dict(torch.load('trainedModel/GRU/trained_encoder1_GRU_hormer_128_M10_Nf_50000.pkl'))
        elif name=="decoder":
            model.load_state_dict(torch.load('trainedModel/GRU/trained_attn_decoder1_GRU_hormer_128_M10_Nf_50000.pkl'))
        elif name=="generator":
            model.load_state_dict(torch.load('trainedModel/GAN/G_LSTM_hormer_12000.pkl'))
        elif name=="discriminator":
            model.load_state_dict(torch.load('trainedModel/GAN/D_LSTM_hormer_12000.pkl'))
        else:
            print "There's no such a model"
            return
    def cuda_use(self):
        return self.use_cuda

class Prepro(Config,object):
    def __init__(self,maxlength,use_cuda):
        super(Prepro,self).__init__(use_cuda)

        self.SOS_token = 0
        self.EOS_token = 1
        self.MAX_LENGTH = maxlength
        self.use_cuda = use_cuda
        self.eng_prefixes=eng_prefixes = (
                            "i am ", "i m ",
                            "he is", "he s ",
                            "she is", "she s",
                            "you are", "you re ",
                            "we are", "we re ",
                            "they are", "they re "
                            )
    def maxlen(self):
        return self.MAX_LENGTH

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def readLangs(self,lang1, lang2,filepath,reverse=False):

        # Read the file and split into lines
        #lines = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')
        lines=open(filepath,encoding='utf-8',errors='ignore').read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[self.normalizeString(s) for s in l.split('#')] for l in lines]
        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            self.input_lang = Lang(lang2)
            self.output_lang = Lang(lang1)
        else:
            self.input_lang = Lang(lang1)
            self.output_lang = Lang(lang2)

        return self.input_lang, self.output_lang, pairs

    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))


    def filterPair(self,p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and \
            len(p[1].split(' ')) < self.MAX_LENGTH #and p[1].startswith(self.eng_prefixes)

    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def prepareData(self,lang1, lang2,filepath,display=True, reverse=False):
        if display:
            input_lang, output_lang, pairs = self.readLangs(lang1, lang2,filepath,reverse)
            print("Read %s sentence pairs" % len(pairs))
            pairs = self.filterPairs(pairs)
            print("Trimmed to %s sentence pairs" % len(pairs))
            print("Counting words...")
            for pair in pairs:
                input_lang.addSentence(pair[0])
                output_lang.addSentence(pair[1])
            print("Counted words:")
            print("Q:", input_lang.n_words)
            print("A:", output_lang.n_words)
            return input_lang, output_lang, pairs
        else:
            input_lang, output_lang, pairs = self.readLangs(lang1, lang2,filepath,reverse)
            pairs = self.filterPairs(pairs)
            for pair in pairs:
                input_lang.addSentence(pair[0])
                output_lang.addSentence(pair[1])
            return input_lang, output_lang, pairs


    def indexesFromSentence(self,lang, sentence):
            return [lang.word2index[word] for word in sentence.split(' ')]

    def variableFromSentence(self,lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def vocChecker(self,lang,sentence):
        voc_error=0
        for word in sentence.split(' '):
            if word not in lang.word2index.keys():
                voc_error=1
        return voc_error

    def variablesFromPair(self,pair):
        input_variable = self.variableFromSentence(self.input_lang, pair[0])
        target_variable = self.variableFromSentence(self.output_lang, pair[1])
        return (input_variable, target_variable)

    def proConv(self,input):
        processed=[]
        input=input.split()
        print input
        for i, word in enumerate(input):
            word=str(word)
            processed.append(unicode(word.lower().translate(None, "?"), "utf-8"))
        return " ".join(processed)

    def words_vectorization(self,model, words):
        line = []
        for j in range(len(words)):
            if words[j] in model.wv.vocab:
                line.append(model[words[j]])
        return np.array(line)