import torch.optim as optim
from model import *
import os
import numpy as np
from preprocessing import *
#from main_tmp import evaluate
import unicodedata
from gensim.models import word2vec

os.environ['CUDA_VISIBLE_DEVICES']='0'
con=Config(use_cuda=False)
pre=Prepro(maxlength=10,use_cuda=False)

def evaluate(encoder, decoder, sentence, max_length=pre.__dict__["MAX_LENGTH"]):
    input_variable = pre.variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[pre.__dict__["SOS_token"]]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoded_indices = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == pre.__dict__["EOS_token"]:
            decoded_words.append('<EOS>')
            decoded_indices.append(ni)
            #print(ni)
            #print(std_output_lang.index2word[ni])
            break
        else:
            #print(ni)
            #print(output_lang.index2word[ni])
            decoded_words.append(std_output_lang.index2word[ni])
            decoded_indices.append(ni)

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return np.array(decoded_words),np.array(decoded_indices), decoder_attentions[:di + 1]

def vec2word(model,vecs):
    words = []
    for i in range(len(vecs)):
         #model.predict(vecs)
        words.append(model.similar_by_vector(vecs[i], topn=1, restrict_vocab=None)[0][0])
    return words


def char_adding(model,G,bot_output_vecs,vector_size):
    g_hidden = G.initHidden()
    g_cell = G.initCell()
    g_inputs=Variable(torch.from_numpy(bot_output_vecs))
    g_outputs = Variable(torch.zeros(len(g_inputs), vector_size)).float()
    #g_outputs = Variable(torch.zeros(pre.__dict__["MAX_LENGTH"], 1))

    for ei in range(len(g_outputs)):
        g_output, (g_hidden, g_cell) = G(g_inputs[ei], g_hidden, g_cell)
        g_outputs[ei] = g_output
    words = vec2word(model,g_outputs.data.numpy())
    return words


def display_bot_out(words):
    encoded=[]
    eliminate_words = set(["<SOS>", "<EOS>"])
    #convert unicode into strings
    for i in range(len(words)):
        #print "jjjj",words[i]
        encoded.append(unicodedata.normalize('NFKD',words[i]).encode('ascii', 'ignore'))
    #eliminate <EOS> <SOS>
    encoded=[word for word in encoded if word not in eliminate_words]
    print "without character: ",
    for i, word in enumerate(encoded):
        print word,
    print
    pass
"""""""""""""""
TEXT DATA LOADING
"""""""""""""""

input_lang, output_lang, pairs = pre.prepareData('que', 'ans',"../data/MovieSuperSmall_homer.txt",False,False)
std_input_lang, std_output_lang, std_pairs = pre.prepareData('que_std', 'ans_std',"../data/MovieSuperSmall.txt",False,False)
char_input_lang, char_output_lang, char_pairs = pre.prepareData('que_char', 'ans_char',"../data/homer_cleaned.txt", False,False)

"""""""""""""""
MODEL DATA LOADING
"""""""""""""""
hidden_size = 128
GAN_hidden_size = 256
z1 = EncoderRNN(input_lang.n_words, hidden_size)#.cuda()
z2 = AttnDecoderRNN(hidden_size,std_output_lang.n_words,1)#.cuda()
G = Generator(input_lang.n_words, GAN_hidden_size,output_lang.n_words)
D = Discriminator(output_lang.n_words,GAN_hidden_size)
model = word2vec.Word2Vec.load("text8.model")

print ("loading pretrained model...")
con.load_parameters(z1,"encoder")
con.load_parameters(z2,"decoder")
con.load_parameters(G,"generator")
#con.load_parameters(D,"discriminator")
print ("loading done")
print

while True:
    input = raw_input("say something")
    if not pre.vocChecker(input_lang,input):
        input = pre.proConv(input)
        bot_output_words, bot_output_indices, attentions = evaluate(z1, z2, input)
        bot_out_vecs = pre.words_vectorization(model,bot_output_words)
        #print "outputtttt",bot_output_words
        display_bot_out(bot_output_words)
        #print bot_output_words
        #print bot_output_indices
        words = char_adding(model,G,bot_out_vecs,GAN_hidden_size)
        print "with character: ",
        print " ".join(words)
    else:
        print("out of vocabulary")
