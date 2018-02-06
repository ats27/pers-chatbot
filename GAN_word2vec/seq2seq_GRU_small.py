# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from preprocessing import *
use_cuda = False
pre=Prepro(maxlength=15,use_cuda=False)

import os
from tensorboardX import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']='0'
SOS_token = 0
EOS_token = 1


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


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def readLangs(lang1, lang2,file, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    #lines = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')
    lines=open(file,encoding='utf-8',errors='ignore').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('#')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and  len(p[1].split(' ')) < MAX_LENGTH #and         p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2,file, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2,file, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print("Q:", input_lang.n_words)
    print("A:", output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang_all,output_lang_all,pairs_all = prepareData('que', 'ans',"../data/normal_goku.txt" ,False)
input_lang, output_lang, pairs = prepareData('que_std', 'ans_std',"../data/normal.txt" ,False)
print(random.choice(pairs))



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair,input_lang,output_lang):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)



teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #input_variable: one sentence represented numerically
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):

        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0][0]


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            print(decoder_output, target_variable[di])
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
            #print(decoder_output)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(10)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]])) #casting the datatype to tensor

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length




import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(pairs,encoder, decoder, n_iters, print_every, save_every=1000, learning_rate=0.01):
    print()
    print('<training start>')
    directory="/home/z_senshi/venv/VDL/chatbot/Hanako/ex3/GAN/log/GRUsmall"+str(n_iters)+"_"+str(MAX_LENGTH)+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("created")
    writer = SummaryWriter(directory)
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs),input_lang,output_lang)for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # iterate for each sentences
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            writer.add_scalar('Train/loss_gru',print_loss_avg , iter)

        if iter % save_every == 0:
            torch.save(encoder1.state_dict(),
                       "trainedModel/GRU/trained_encoder1_GRU_hormer_" + str(hidden_size) + "_M" + str(MAX_LENGTH) + "_Nf_" + str(
                           iter) + ".pkl")
            torch.save(attn_decoder1.state_dict(),
                       "trainedModel/GRU/trained_attn_decoder1_GRU_hormer_" + str(hidden_size) + "_M" + str(MAX_LENGTH) + "_Nf_" + str(
                           iter) + ".pkl")


    writer.close()

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            #print(ni)
            #print(output_lang.index2word[ni])
            break
        else:
            print(ni)
            print(output_lang.index2word[ni])
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]




def evaluateRandomly(encoder, decoder,input, n=10):
    print(input)
    output_words, attentions = evaluate(encoder, decoder, input)
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    """

hidden_size = 128

encoder1 = EncoderRNN(input_lang_all.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang_all.n_words,1, dropout_p=0.1)#hidden_size of decoder has to equal to number of vocabulary(output_lang.n_words).
#con=Config(use_cuda=False)
#print ("loading pretrained model")
#con.load_parameters(encoder1,"encoder")
#con.load_parameters(attn_decoder1,"decoder")
#print ("loading done")

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
#print ("loading pretrained model")
#encoder1.load_state_dict(torch.load('trained_encoder1.pkl'))
#attn_decoder1.load_state_dict(torch.load('trained_attn_decoder1.pkl'))
#print ("loading done")
iteration=2000

trainIters(pairs,encoder1, attn_decoder1, iteration, print_every=100,save_every=500)
torch.save(encoder1.state_dict(),"trainedModel/GRU/trained_encoder1_GRU_hormer_"+str(hidden_size)+"_M"+str(MAX_LENGTH)+"_Nf_"+str(iteration)+".pkl")
torch.save(attn_decoder1.state_dict(),"trainedModel/GRU/trained_attn_decoder1_GRU_hormer_"+str(hidden_size)+"_M"+str(MAX_LENGTH)+"_Nf_"+str(iteration)+".pkl")
######################################################################

input = pre.proConv("do you know how much time i used")
#print(input)
evaluateRandomly(encoder1, attn_decoder1,input)

