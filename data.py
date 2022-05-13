#%%
import io
import json
import nltk
import re
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from constant import *

nltk.download('stopwords')
nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('omw-1.4')


class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer_save = None
        self.input_data = None
        self.labels_data = None

    def split_data(self, sentences, labels, test_size=0.2):
        """Split data into training and test sets"""
        X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=test_size, random_state=42)
        return X_train, X_val, y_train, y_val

    def encode_labels(self, data):
        """Encode labels to numbers"""
        dt_unique = data.unique()
        return np.array(data.replace(dt_unique,[i for i in range(len(dt_unique))]))

    def remove_html_tags(self, text):
        """Remove html tags from a text"""
        clean_text = re.compile(r'<.*?>')
        return clean_text.sub(r'', text)

    def remove_url(self, text):
        """Remove url from a text"""
        clean_text = re.compile(r'https?://\S+|www\.\S+')
        return clean_text.sub(r'', text)

    def remove_punctuation(self, text):
        """Remove punctuation from a text"""
        exclude = string.punctuation
        for char in exclude:
            text = text.replace(char,'')
        return text

    def remove_emoji(self, text):
        """Remove emoji from a text"""
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_stopwords(self, text):
        """Remove stopwords from a text"""
        stop_words=set(stopwords.words('english'))
        words = text.split()
        list_words = []
        for word in words:
            if word not in stop_words:
                list_words.append(word)
        return ' '.join(list_words)

    #grouping together the different inflected forms of a word
    def lemma_traincorpus(self, data):
        """Lemmatize the training corpus"""
        lemmatizer=WordNetLemmatizer()
        sentence_words = word_tokenize(data)
        out_data=[lemmatizer.lemmatize(word, pos='v') for word in sentence_words]
        return ' '.join(out_data)

    def tokenizer_data(self, sentences, vocab_size, max_length):
        """Tokenize and pad sentences"""
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=OOV)
        tokenizer.fit_on_texts(sentences)
        self.tokenizer_save = tokenizer
        # Generate and pad the training sequences
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=max_length, truncating=TRUNC_TYPE, padding=PADDING)
        return padded

    def save_clean_data(self, sentences, list_labels, input, label):
        """Save clean data to a file"""
        # list_sentences = [' '.join(sentence) for sentence in sentences]
        output = pd.DataFrame({input: sentences,label: list_labels})
        output.to_csv('./data/clean/clean_data.csv', index=False)
        print("Clean data saved.")

    def save_tokenizer(self, tokenizer):
        """Save tokenizer to a file"""
        tokenizer_json = tokenizer.to_json()
        with io.open('./data/tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        print("Tokenizer saved.")

    def save_labels(self, data):
        """Save labels to a file"""
        dt_unique = data.unique()
        self.labels_data = dict(zip(dt_unique, [i for i in range(len(dt_unique))]))
        with open('./data/label.json', 'w') as f:
            json.dump(self.labels_data, f)
        print("Labels saved.")

    def clean_data(self, texts):
        """Clean the data by removing html tags, url, 
        punctuation, stopwords, emoji, lemmatize"""
        texts = texts.str.lower().str.strip()
        for i in tqdm(range(len(texts))):
            texts[i] = self.remove_html_tags(texts[i])
            texts[i] = self.remove_url(texts[i])
            texts[i] = self.remove_emoji(texts[i])
            texts[i] = self.remove_punctuation(texts[i])
            texts[i] = self.remove_stopwords(texts[i])
            texts[i] = self.lemma_traincorpus(texts[i])
        return np.array(texts)

    def load_dataset(self, max_length, vocab_size, input_name, label_name, cleaned_data):
        """Load and preprosess the dataset"""
        # Load, clean the dataset
        print("Loading dataset...")
        data = pd.read_csv(self.data_path)
        data = data.dropna()
        labels = self.encode_labels(data[label_name])
        if cleaned_data:
            sentences = np.array(data[input_name])
        else:
            sentences = self.clean_data(data[input_name])
            # Save data after preprocessing
            self.save_clean_data(sentences, labels, input_name, label_name)
            self.save_labels(data[label_name])
        padded_sentences = self.tokenizer_data(sentences, vocab_size, max_length)
        self.save_tokenizer(self.tokenizer_save)
        print("Dataset loaded.")
        return padded_sentences, labels

    def build_dataset(self, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, test_size=TEST_SIZE, buffer_size=128,
                      batch_size=4, input_name=INPUT_NAME, label_name=LABEL_NAME, cleaned_data=False):
        """Build the dataset"""
        padded_sentences, labels = self.load_dataset(max_length, vocab_size, input_name, label_name, cleaned_data)
        X_train, X_val, y_train, y_val = self.split_data(padded_sentences, labels, test_size)
        print(X_train.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_train, dtype=tf.int64),
                                                            tf.convert_to_tensor(y_train, dtype=tf.int64)))
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_val, dtype=tf.int64),
                                                          tf.convert_to_tensor(y_val, dtype=tf.int64)))
        val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)
        return train_dataset, val_dataset

if __name__ == "__main__":
    data_path = '/home/whoisltd/Documents/Transformer-XL/data/clean/clean_data.csv'
    data_clean = Dataset(data_path)
    a , b = data_clean.build_dataset(cleaned_data=True)
    # print(a)
    for (batch, (inputs, labels)) in enumerate(a):
        if batch == 0:
            print(inputs.shape)
