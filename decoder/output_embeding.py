#input embeding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

"""
ROW CODE BY ME
class input_encoding:
    def __init__(self, sentence):
        self.sent = sentence
        self.pre_process()
        self.size = len(self.words)
        self.input_vector = []
        self.creat_vector
        return self.input_vector

    def pre_process(self):
        stop_words = set(stopwords.words('english'))
        words_token = word_tokenize(self.sent)
        self.words = []
        for word in words_token:
            if word not in stop_words:self.words.append(word)

    def creat_vector(self):
        lis = [0] * self.size
        for i in range(self.size):
            lis[i] = 1
            self.input_vector.append(lis)
            lis[i] = 0
"""

#converting each word in a sentence to 1 X 512 dimension vector 

class OutputEmbedding:
    def __init__(self, sentence = ['my name is saurab singh']):
        self.input = sentence
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentence)
        self.word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(sentence)
        self.padded_sequences = pad_sequences(sequences)
        self.process()
    
    def process(self):
        embed_layer = Embedding(input_dim = len(self.word_index) + 1, output_dim = 512, input_length=self.padded_sequences.shape[1])
        self.embedding_sequences = embed_layer(self.padded_sequences)
    
    def get_encoded_vector(self):
        return self.embedding_sequences
     