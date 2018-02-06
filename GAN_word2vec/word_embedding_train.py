
import logging
from gensim.models import word2vec
# show progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# prepare dataset for training
sentences = word2vec.Text8Corpus('text8')

# create Word2Vec instance
# sentences : sentences to be processed
# size      : lenght of word vector
# min_count : ignore word which appear less times than this value
# window    : check the back and forword words with this window size
model = word2vec.Word2Vec(sentences, size=256, min_count=1, window=15)

# save the model
model.save("text8.model")

if __name__ == '__main__':
    print "Finish!!!"