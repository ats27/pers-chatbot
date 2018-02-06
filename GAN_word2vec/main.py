import torch.optim as optim
from model import *
import os
import numpy as np
import random
from torch import optim
from tensorboardX import SummaryWriter
from preprocessing import *
from gensim.models import word2vec

os.environ['CUDA_VISIBLE_DEVICES']='0'
con=Config(use_cuda=False)
pre=Prepro(maxlength=10,use_cuda=False)

def trainIters(model,vector_size,G,D, pairs,criterion,n_iters,z1,z2, print_every=500, plot_every=100, learning_rate=0.001):
    print('<training start>')
    writerG = SummaryWriter('log/GAN/G_loss/')
    writerD = SummaryWriter('log/GAN/D_loss/')
    writerD_real = SummaryWriter('log/GAN/D_real_loss/')
    writerD_fake = SummaryWriter('log/GAN/D_fake_loss/')
    d_fake_loss = 0
    d_real_loss = 0
    g_loss=0
    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):
        # iterate for each sentences
        pair=random.choice(pairs)
        question=pair[0]
        gt_answer=pair[1]
        gt_vecs = pre.words_vectorization(model, gt_answer.split())
        noise_words, noise_indices, attentions = evaluate(z1, z2,question)
        noise_vecs=pre.words_vectorization(model,noise_words)
        if len(gt_vecs)*len(noise_vecs)!=0:
            gt_vecs=Variable(torch.from_numpy(gt_vecs))
            noise_vecs=Variable(torch.from_numpy(noise_vecs))
            """""""""""""""
            Discriminator Training
            """""""""""""""
            g_hidden = G.initHidden()
            g_cell = G.initCell()
            d_hidden = D.initHidden()
            d_cell = D.initCell()
            D.zero_grad()
            #  1A: Train D on real
            d_real_data = gt_vecs
            for ei in range(len(d_real_data)):
                d_output, (d_hidden, d_cell) = D(d_real_data[ei],d_hidden, d_cell)
            d_real_decision=d_output
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
            d_real_error.backward(retain_graph=True)  # compute/store gradients, but don't change params

            gen_input = noise_vecs
            g_outputs = Variable(torch.zeros(len(gen_input),vector_size)).float()
            for ei in range(len(gen_input)): # need to add detatch() later
                g_output, (g_hidden,g_cell) = G(gen_input[ei],g_hidden,g_cell)#.detach()
                g_outputs[ei] = g_output
            d_fake_data = g_outputs  # detach to avoid training G on these labels

            for ei in range(len(d_fake_data)):

                d_output, (d_hidden, d_cell) = D(d_fake_data[ei] ,d_hidden, d_cell)

            d_fake_decision=d_output
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))#.cuda())  # zeros = fake
            d_fake_error.backward(retain_graph=True)
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

            d_real_loss += d_real_error.data[0]
            d_fake_loss += d_fake_error.data[0]

            """""""""""""""
            Generator Training
            """""""""""""""
            for k in range(2):
                g_hidden = G.initHidden()
                g_cell = G.initCell()
                d_hidden = D.initHidden()
                d_cell = D.initCell()
                G.zero_grad()

                gen_input = noise_vecs#.cuda()
                g_outputs = Variable(torch.zeros(len(gen_input),vector_size)).float()
                #feed G giving single word
                for ei in range(len(gen_input)):
                        g_output, (g_hidden,g_cell) = G(gen_input[ei],g_hidden,g_cell)
                        g_outputs[ei] = g_output

                for ei in range(len(g_outputs)):
                        d_output, (d_hidden, d_cell) = D(g_outputs[ei], d_hidden, d_cell)


                dg_fake_decision = d_output
                g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))
                g_error.backward(retain_graph=True)
                g_optimizer.step()  # Only optimizes G's parameters

            g_loss += g_error.data[0]

            if iter % print_every == 0:
                print('Iter:{} G_Loss:{:.5f}  D_Loss:{:.5f} (D_real_Loss{:.5f}  D_fake_loss{:.5f})'.format(
                    iter,
                    g_loss,
                    (d_real_loss+d_fake_loss)/float(2),
                    d_real_loss,
                    d_fake_loss
                ))
                writerG.add_scalar('loss', g_loss, iter)
                writerD.add_scalar('loss', (d_real_loss+d_fake_loss)/float(2), iter)
                writerD_fake.add_scalar('loss',  d_fake_loss, iter)
                writerD_real.add_scalar('loss',  d_real_loss, iter)
                torch.save(G.state_dict(), "trainedModel/GAN/G_LSTM_hormer_" + str(iter) + ".pkl")
                torch.save(D.state_dict(), "trainedModel/GAN/D_LSTM_hormer_" + str(iter) + ".pkl")
                g_loss=0
                d_real_loss=0
                d_fake_loss=0

    writerG.close()
    writerD.close()
    writerD_fake.close()
    writerD_real.close()

def evaluate(encoder, decoder, sentence, max_length=pre.__dict__["MAX_LENGTH"]):
    input_variable = pre.variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([pre.__dict__["SOS_token"]]))  # SOS
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
            break
        else:
            decoded_words.append(std_output_lang.index2word[ni])
            decoded_indices.append(ni)
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return np.array(decoded_words),np.array(decoded_indices), decoder_attentions[:di + 1]


"""""""""""""""
TEXT DATA LOADING
"""""""""""""""

input_lang, output_lang, pairs = pre.prepareData('que', 'ans',"../data/MovieSuperSmall_homer.txt",True,False)
std_input_lang, std_output_lang, std_pairs = pre.prepareData('que_std', 'ans_std',"../data/MovieSuperSmall.txt",True,False)
char_input_lang, char_output_lang, char_pairs = pre.prepareData('que_char', 'ans_char',"../data/homer_cleaned.txt",True,False)

"""""""""""""""
MODEL DATA LOADING
"""""""""""""""
hidden_size = 128
z1 = EncoderRNN(input_lang.n_words, hidden_size)#.cuda()
z2 = AttnDecoderRNN(hidden_size, std_output_lang.n_words,1, dropout_p=0.1)#.cuda()
print ("loading pretrained model")
con.load_parameters(z1,"encoder")
con.load_parameters(z2,"decoder")
print ("loading done")
GAN_hidden_size = 256
#output_words,output_indices,attentions = evaluate(z1,z2, char_pairs[0][0])
G = Generator(input_lang.n_words, GAN_hidden_size,output_lang.n_words)
D = Discriminator(output_lang.n_words,GAN_hidden_size)
criterion = nn.BCELoss()  # Binary cross entropy

if use_cuda:
    z1 = z1.cuda()
    z2 = z2.cuda()

model = word2vec.Word2Vec.load("text8.model")
iter=10000
trainIters(model,256,G,D,char_pairs,criterion,iter,z1,z2)
torch.save(G.state_dict(),"trainedModel/GAN/G_LSTM_hormer_"+str(iter)+".pkl")
torch.save(D.state_dict(),"trainedModel/GAN/D_LSTM_hormer_"+str(iter)+".pkl")