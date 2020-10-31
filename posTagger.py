import pandas as pd
import re
import nltk
import copy
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

class posTagger:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')
        
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
            if (lemma not in self.stopwords and len(lemma)>1):
                temp.append(lemma)
        tokens = copy.copy(temp)
        
        return tokens
    
    def execute(self, file_name):
        data = pd.read_csv(file_name)
        training_data = data['review']
        training_target = data['rating']

        tagged_result = []
        for review in training_data:
            tagged = nltk.pos_tag(self.preprocess(review))
            tagged_result.append(tagged)
            print(tagged)

if __name__ == "__main__":
    posTag = posTagger()
    posTag.execute("data/data.csv")