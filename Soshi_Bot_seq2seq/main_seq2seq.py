import numpy as np
from data_vectorization import sentence_vectorization
import pickle
import numpy as np
import os
from keras.models import Sequential
import gensim
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
import theano
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def sentence_len_equalizer(length,sentences):
    lines = []
    for (i, sentence) in enumerate(sentences):
        lines.append(sentence.split(" "))

    for line in lines:
        line[length-1:] = []

        if len(line) < length:
            for i in range(length - len(line)):
                line.append("<PAD>")
    return np.array(lines)

def sentence_len_equalizer_word2vec(length,sentences):
    lines = []
    for sentence in sentences:
        sentence[length-1:] = []

        if len(sentence) < length:
            for i in range(length - len(sentence)):
                sentence.append(np.ones(30,dtype=np.float32))
    return np.array(sentences)

# Load the data
answers = open('answer.txt').read().split('\n')
questions = open('question.txt').read().split('\n')

#answers=sentence_len_equalizer(15,answers).reshape((138333, 15,1))
#questions=sentence_len_equalizer(15,qu estions).reshape((138333, 15,1))

answers= sentence_vectorization(answers,"text8.model")
answers= sentence_len_equalizer_word2vec(15,answers)
questions= sentence_vectorization(questions,"text8.model")
questions= sentence_len_equalizer_word2vec(15,questions)


theano.config.optimizer = "None"


vec_x = np.array(questions, dtype=np.float64)
vec_y = np.array(answers, dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)

print x_test.shape
n_hidden=128
n_out=30
model = Sequential()
model.add(LSTM(n_hidden, input_shape=x_train.shape[1:]))

model.add(RepeatVector(15))
model.add(LSTM(n_hidden,return_sequences=True))

model.add(TimeDistributed(Dense(n_out, activation='softmax')))

print model.input_shape
print model.output_shape
print model.summary()

model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=1,validation_data=(x_test, y_test))
model.save('LSTM50.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM100.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM150.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM200.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM250.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM300.h5');
model.fit(x_train, y_train, nb_epoch=50,validation_data=(x_test, y_test))
model.save('LSTM350.h5');

predictions = model.predict(x_test)
mod = gensim.models.Word2Vec.load("text8.model");
[mod.most_similar([predictions[10][i]])[0] for i in range(15)]
