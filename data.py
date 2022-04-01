from base64 import encode
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import time
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
class Dataset:
    def __init__(self, data_path, ):
        self.data_path = data_path
        self.vocab_size = 10000
        self.max_length = 120
        self.embedding_dim = 16
        self.trunc_type='post'
        self.oov_tok = "<OOV>"

    def load_dataset(self):
        print("Loading dataset...")
        data = pd.read_csv(self.data_path)
        data = data.dropna()
        data['review'] = data['review'].str.lower()
        labels = list(self.encode_labels(data['sentiment']))
        sentences = list(self.clean_data(data['review']))
        sentences = self.tokenizer_data(sentences, self.vocab_size, self.max_length, self.trunc_type, self.oov_tok)
        return sentences, labels

    def split_data(self, sentences, labels, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=test_size, random_state=42)
        return X_train, X_val, y_train, y_val

    def build_dataset(self):
        sentences, labels = self.load_dataset()
        X_train, X_val, y_train, y_val = self.split_data(sentences, labels)
        print(type(X_train))
        return ((X_train, X_val), (y_train, y_val))
    
    def clean_data(self, texts):
        for i in range(len(texts)):
            texts[i] = self.remove_html_tags(texts[i])
            texts[i] = self.remove_url(texts[i])
            texts[i] = self.remove_emoji(texts[i])
            texts[i] = self.remove_punctuation(texts[i])
            texts[i] = self.convert_incorrect_text(texts[i])
            texts[i] = self.remove_stopwords(texts[i])
            texts[i] = self.lemma_traincorpus(texts[i])
        return texts

    def encode_labels(self, df):
        df['sentiment'].replace({'negative':0,'positive':1}, inplace=True)

    def remove_html_tags(self, text):
        clean_text = re.compile(r'<.*?>')
        return clean_text.sub(r'', text)

    def remove_url(self, text):
        clean_text = re.compile(r'https?://\S+|www\.\S+')
        return clean_text.sub(r'', text)

    def remove_punctuation(text):
        exclude = string.punctuation
        for char in exclude:
            text = text.replace(char,'')
        return text

    def convert_incorrect_text(self, text):
        correct = TextBlob(text)
        return correct.correct().string
    
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        words = text.split()
        list_words = []
        for word in words:
            if word not in stop_words:
                list_words.append(word)
        return ' '.join(list_words)
    
    def lemma_traincorpus(data):
        lemmatizer=WordNetLemmatizer()
        out_data=[lemmatizer.lemmatize(word) for word in data]
        return out_data

    def tokenizer_data(sentences, vocab_size, max_length, trunc_type, oov_tok):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        # Generate the word index dictionary for the training sentences
        tokenizer.fit_on_texts(sentences)
        # word_index = tokenizer.word_index
        # Generate and pad the training sequences
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
        return padded

    