from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
use_cuda = False

###### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_layers=1):
        super(Generator, self).__init__()
        self.n_layers = n_layers
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

    def forward(self, input, hidden, cell):
        # for i in range(self.n_layers):
        #    output, hidden = self.gru(output,(hidden,cell))

       # embedded = self.embedding(input).view(1, 1, -1)
        output = input.view(1,1, self.hidden_size) #embedded.view(1,self.hidden_size)
        output, hidden = self.lstm(output, (hidden, cell))  # unlike GRU, LSTM need cell state too. (google it). and returns output,(hidden, cell)
        #output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def initCell(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size,output_size=1, n_layers=1):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        # for i in range(self.n_layers):
        #    output, hidden = self.gru(output,(hidden,cell))
        #embedded = self.embedding(input).view(1, 1, -1)
        output = input.view(1,1, self.hidden_size) #embedded.view(1, self.hidden_size)
        output, hidden = self.lstm(output, (hidden, cell))  # unlike GRU, LSTM need cell state too. (google it). and returns output,(hidden, cell)
        output =  F.sigmoid(self.out(output))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def initCell(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
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

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

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
