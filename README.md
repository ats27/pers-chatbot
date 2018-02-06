# Chat bot with personality

### Brief Explanation for each Files
- **Soshi_Bot1** - chatbot with LSTM, fixed length <br />
- **movie_lines.txt** - raw dataset which contains scripts from various movies <br />
- **movie_conversations.txt** - raw dataset which anotates corresponding conversations <br />
- **data_preprocessing.py** - data preprocessing which generate answers and questions file. <br />Each lines in both files correspond.<br />
- **word_embedding_train.py** - training for word2vec using text8, and generate text8.model<br />
- **answers.txt** - answers generated by data_preprocessing.py<br /> 
- **questions.txt** - questions generated by data_preprocessing.py<br /> 
- **text8** - dataset for word2vec training
- **text8.model** - trained model for word embedding
- **data_vectorization.py** - apply word embedding for prepared answers and questions <br />
### Prerequisites
to be continued...
