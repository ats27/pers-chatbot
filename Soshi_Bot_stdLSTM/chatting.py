import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import theano


theano.config.optimizer = "None"
model = load_model('LSTM500.h5')
mod = gensim.models.Word2Vec.load("text8.model");

while (True):
    x = raw_input("Enter the message:");
    sentend = np.ones((30L,), dtype=np.float32)

    sent = nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.wv.vocab]

    sentvec[14:] = []
    sentvec.append(sentend)
    if len(sentvec) < 15:
        for i in range(15 - len(sentvec)):
            sentvec.append(sentend)
    sentvec = np.array([sentvec])
    predictions = model.predict(sentvec)
    outputlist = [mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    output = ' '.join(outputlist)
    print output
