import numpy as np
import pickle
import os
from nltk.corpus import wordnet, brown, treebank, conll2000
from keras.models import Sequential, Model, load_model
from keras.layers import (
    InputLayer, 
    LSTM, 
    Embedding, 
    TimeDistributed, 
    Dense, 
    Bidirectional, 
    Activation,
    Dropout
)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import backend

from sklearn.model_selection import train_test_split

class posTagger:            
    def sequences_to_tags(self, predictions, tag_map):
        tag_result = []
        for prediction in predictions:
            not_padding = False
            tag_list = []
            for index in prediction:
                tag = tag_map[np.argmax(index)]
                if (tag != "<<PAD>>"):
                    not_padding = True
                if (not_padding):
                    tag_list.append(tag)
    
            tag_result.append(tag_list)
    
        return tag_result
    
    def load(self, path):
        with open(path, 'rb') as f:
            word2int, tag2int, MAX_LENGTH = pickle.load(f)
            return word2int, tag2int, MAX_LENGTH
    
    def pos_tag(self, token_list):
        bi_lstm_model = load_model("model/bi_lstm_model.h5")
        word_tokenizer, tag_tokenizer, MAX_LENGTH = self.load('PickledData/data.pkl')
        
        input_sequences = word_tokenizer.texts_to_sequences(token_list)
        input_sequences = pad_sequences(input_sequences, maxlen=MAX_LENGTH, padding='pre')
        predictions = bi_lstm_model.predict(input_sequences)

        reverse_tag_map = dict(map(reversed, tag_tokenizer.word_index.items()))
        tag_result = self.sequences_to_tags(predictions, reverse_tag_map)

        result = []
        for i in range(len(token_list)):
            if (len(token_list[i]) != len(tag_result[i])):
                diff = len(token_list[i]) - len(tag_result[i])
                if (diff > 0):
                    for j in range(diff):
                        tag_result[i].insert(0, 'NOUN')
            result.append(list(zip(token_list[i], tag_result[i])))
        
        return result

    def execute(self):
        treebank_corpus = treebank.tagged_sents(tagset='universal')
        brown_corpus = brown.tagged_sents(tagset='universal')
        conll_corpus = conll2000.tagged_sents(tagset='universal')
        tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

        TEST_SIZE = 0.1
        VAL_SIZE = 0.15
        EPOCH_COUNT = 3
        BATCH_SIZE = 128

        X = []
        Y = []

        for sentence in tagged_sentences:
            words_temp = []
            tags_temp = []
            for pair in sentence:         
                words_temp.append(pair[0])
                tags_temp.append(pair[1])
            X.append(words_temp)
            Y.append(tags_temp)

        words = set([word.lower() for sentence in X for word in sentence])
        tags = set([tag for sentence in Y for tag in sentence])

        word_tokenizer = Tokenizer(lower=True, oov_token='<<OOV>>')
        word_tokenizer.fit_on_texts(X)
        X_sequence = word_tokenizer.texts_to_sequences(X)

        tag_tokenizer = Tokenizer(lower=False)
        tag_tokenizer.fit_on_texts(Y)
        Y_sequence = tag_tokenizer.texts_to_sequences(Y)
        tag_tokenizer.word_index['<<PAD>>'] = 0

        X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_sequence, Y_sequence, test_size=TEST_SIZE)

        MAX_LENGTH = len(max(X_train_, key=len))
        X_train_ = pad_sequences(X_train_, maxlen=MAX_LENGTH, padding='pre')
        X_test_ = pad_sequences(X_test_, maxlen=MAX_LENGTH, padding='pre')
        Y_train_ = pad_sequences(Y_train_, maxlen=MAX_LENGTH, padding='pre')
        Y_test_ = pad_sequences(Y_test_, maxlen=MAX_LENGTH, padding='pre')

        Y_train_ = to_categorical(Y_train_)

        bi_lstm_model = Sequential()
        bi_lstm_model.add(InputLayer(input_shape=(MAX_LENGTH,)))
        bi_lstm_model.add(Embedding(len(word_tokenizer.word_index), 128))
        bi_lstm_model.add(Bidirectional(LSTM(256, return_sequences=True)))
        bi_lstm_model.add(Dropout(0.1))
        bi_lstm_model.add(TimeDistributed(Dense(len(tag_tokenizer.word_index))))
        bi_lstm_model.add(Activation('softmax'))
        bi_lstm_model.summary()

        bi_lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

        bi_lstm_model.fit(X_train_, Y_train_, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, validation_split=VAL_SIZE)

        bi_lstm_model.save("model/bi_lstm_model.h5")

        pickle_files = [word_tokenizer, tag_tokenizer, MAX_LENGTH]

        if not os.path.exists('PickledData/'):
            os.makedirs('PickledData/')

        with open('PickledData/data.pkl', 'wb') as f:
            pickle.dump(pickle_files, f)

if __name__ == "__main__":
    test_samples = [
        "running is very important for me .".split(),
        "I was running every day for a month .".split(),
        "like games get garry s mod everything like games pretty much garry s mod nuff said".split(),
        "way game warped way modding community community kept game alive years forced turn ruined game longer exist innocent modding community community built solely mutual respect love game hobby good faith recommend game anyone current state nothing positive supersede massive negative implemented valve".split()
    ]

    test_samples = [
        ['skyrim', 'nt', 'good', 'game', 'without', 'mods', 'fact', 'might', 'pay', 'mods', 'make', 'bugthesda', 's', 'game', 'playable', 'rubbish'],
        ['addictive', 'game', 'ever', 'made'],
        ['counter', 'strike', 'even', 'fight', 'highly', 'trained', 'american', 'antiterrorist', 'team', 'using', 'latest', 'military', 'technology', 'battle', 'group', 'really', 'madmen', 'possessing', 'crude', 'bomb', 'surplus', 'ussr', 's', 'army', 'supplies', 'despite', 'training', 'technology', 'terrorists', 'still', 'good', 'chance', 'blowing', 'market', 'therefore', 'much', 'like', 'real', 'life', 'game', 'currently', 'full', 'hackers', 'fly', 'top', 'map', 'unless', 'hack', 'like', 'getting', 'aerial', 'teabag', 'please', 'play', 'better', 'counter', 'strike', 'sauce', 'counter', 'strike', 'go', 'game', 'game', 'day', 'exists', 'historical', 'purposes', 'remember', 'times', 'internet', 'cafe', 'mosque', 'full', 'game']
    ]

    for i in test_samples:
        print(len(i))

    posTag = posTagger()
    # posTag.execute()
    result = posTag.pos_tag(test_samples)
    for res in result:
        print(res)
        print(len(res))