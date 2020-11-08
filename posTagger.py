import pandas as pd
import numpy as np
import re
import nltk
import copy
import pickle
import os
from nltk.corpus import stopwords, wordnet, brown, treebank, conll2000
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, LSTM, Embedding, TimeDistributed, Dense, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras import backend

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from gensim.models import KeyedVectors

class posTagger:        
    def clean_non_char(self, sentence):
        #replace all non characters with whitespace
        cleaned = re.sub("[^A-Za-z]+", " ", sentence)

        #replace duplicate whitespace into one whitespace
        ' '.join(cleaned.split())
        
        #case folding and remove header and trailer whitespace
        return cleaned.lower().strip()

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, sentence):
        lowercase_only_text = self.clean_non_char(sentence)
        
        #tokenization
        tokens = nltk.word_tokenize(lowercase_only_text)
        
        #lemmatization
        temp = []
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))
            
            #filtering
            # if (lemma not in self.stopwords and len(lemma)>1):
            #     temp.append(lemma)
        tokens = copy.copy(temp)
        
        return tokens
    
    def to_categorical(self, sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)
    
    def logits_to_tokens(self, sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                tag = index[np.argmax(categorical)]
                if (tag == "-PAD-"):
                    break
                else:
                    token_sequence.append(tag)
    
            token_sequences.append(token_sequence)
    
        return token_sequences
    
    def load(self, path):
        with open(path, 'rb') as f:
            word2int, tag2int, MAX_LENGTH = pickle.load(f)
            return word2int, tag2int, MAX_LENGTH
    
    def pos_tag(self, token_list):
        bi_lstm_model = load_model("model/bi_lstm_model.h5")
        word2int, tag2int, MAX_LENGTH = self.load('PickledData/data.pkl')

        input_sequences = []
        for sentence in token_list:
            sentence_int = []
            for word in sentence:
                try:
                    sentence_int.append(word2int[word.lower()])
                except:
                    sentence_int.append(word2int['-OOV-'])
            input_sequences.append(sentence_int)
        
        input_sequences = pad_sequences(input_sequences, maxlen=MAX_LENGTH, padding='post')
        
        predictions = bi_lstm_model.predict(input_sequences)
        return self.logits_to_tokens(predictions, {i: t for t, i in tag2int.items()})

    def execute(self):
        # data = pd.read_csv(file_name)
        # training_data = data['review']
        # training_target = data['rating']

        # tagged_result = []
        # for review in training_data:
        #     tagged = nltk.pos_tag(self.preprocess(review))
        #     tagged_result.append(tagged)

        treebank_corpus = treebank.tagged_sents(tagset='universal')
        brown_corpus = brown.tagged_sents(tagset='universal')
        conll_corpus = conll2000.tagged_sents(tagset='universal')
        tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

        # words = []
        # tags = []
        X = []
        Y = []

        for sentence in tagged_sentences:
            words_temp = []
            tags_temp = []
            for pair in sentence:         
                words_temp.append(pair[0])
                # words.append(pair[0])
                tags_temp.append(pair[1])
                # tags.append(pair[1])
            X.append(words_temp)
            Y.append(tags_temp)

        # words = set(words)
        # tags = set(tags)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        words = set([])
        tags = set([])

        for sentence in X_train:
            for word in sentence:
                words.add(word.lower())
        
        for sentence_tags in Y_train:
            for tag in sentence_tags:
                tags.add(tag)
            
        word2int = {w: i + 2 for i, w in enumerate(list(words))}
        word2int['-PAD-'] = 0
        word2int['-OOV-'] = 1
        
        tag2int = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2int['-PAD-'] = 0

        X_train_ = []
        X_test_ = []
        Y_train_ = []
        Y_test_ = []

        for sentence in X_train:
            sentence_int = []
            for word in sentence:
                try:
                    sentence_int.append(word2int[word.lower()])
                except:
                    sentence_int.append(word2int['-OOV-'])
            X_train_.append(sentence_int)
        
        for sentence in X_test:
            sentence_int = []
            for word in sentence:
                try:
                    sentence_int.append(word2int[w.lower()])
                except:
                    sentence_int.append(word2int['-OOV-'])
            X_test_.append(sentence_int)
        
        for tag in Y_train:
            Y_train_.append([tag2int[i] for i in tag])
        
        for tag in Y_test:
            Y_test_.append([tag2int[i] for i in tag])

        MAX_LENGTH = len(max(X_train_, key=len))
        X_train_ = pad_sequences(X_train_, maxlen=MAX_LENGTH, padding='post')
        X_test_ = pad_sequences(X_test_, maxlen=MAX_LENGTH, padding='post')
        Y_train_ = pad_sequences(Y_train_, maxlen=MAX_LENGTH, padding='post')
        X_test_ = pad_sequences(X_test_, maxlen=MAX_LENGTH, padding='post')

        bi_lstm_model = Sequential()
        bi_lstm_model.add(InputLayer(input_shape=(MAX_LENGTH,)))
        bi_lstm_model.add(Embedding(len(word2int), 128))
        bi_lstm_model.add(Bidirectional(LSTM(256, return_sequences=True)))
        bi_lstm_model.add(TimeDistributed(Dense(len(tag2int))))
        bi_lstm_model.add(Activation('softmax'))
        bi_lstm_model.summary()

        bi_lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

        bi_lstm_model.fit(X_train_, self.to_categorical(Y_train_, len(tag2int)), batch_size=128, epochs=3, validation_split=0.2)

        bi_lstm_model.save("model/bi_lstm_model.h5")

        pickle_files = [word2int, tag2int, MAX_LENGTH]

        if not os.path.exists('PickledData/'):
            os.makedirs('PickledData/')

        with open('PickledData/data.pkl', 'wb') as f:
            pickle.dump(pickle_files, f)

if __name__ == "__main__":
    test_samples = [
        "running is very important for me .".split(),
        "I was running every day for a month .".split()
    ]

    # print(test_samples)
    posTag = posTagger()
    # posTag.execute()
    print(posTag.pos_tag(test_samples))