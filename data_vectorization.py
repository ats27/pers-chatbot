from gensim.models import word2vec

"""
function which vectorize dataset
"""
def sentence_vectorization(sentences,model_name):

    # Load the trained model
    model = word2vec.Word2Vec.load(model_name)

    line = []
    for (i,sentence) in enumerate(sentences):
        temp = []
        split_sentence = sentence.split(" ")
        for j in range(len(split_sentence)):
            if split_sentence[j] in model.wv.vocab:
                temp.append(model[split_sentence[j]])
        line.append(temp)
    return line


"""
Example 
"""
# Load the data
answers = open('answer.txt').read().split('\n')
questions = open('question.txt').read().split('\n')

print len(answers)
print len(questions)

answers_vec = sentence_vectorization(answers,"text8.model")
question_vec = sentence_vectorization(questions,"text8.model")

print len(answers_vec)       #shows number of answers
print len(answers_vec[0])    #shows number of words of the sentence
print len(answers_vec[0][0]) #specified length of a word vec
print answers_vec[0][0]      #vector representation of a word

