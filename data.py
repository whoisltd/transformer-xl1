import tensorflow as tf
import numpy as np
import pandas as pd
import string
import time
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
class Dataset:
    def __init__(self, data_path, ):
        self.data_path = data_path
        pass

    def load_dataset(self, ):
        data = pd.read_csv(self.data_path)
        data = data.dropna()
        data['reviewer'] = data['reviewer'].str.lower()

    def split_data(self):
        data = shuffle(data)
        train_data, test_data = train_test_split(data, test_size=0.2)
        return train_data, test_data

    def build_dataset(self):
        pass
    
    def remove_html_tags(self, text):
        clean_text = re.compile(r'<.*?>')
        return clean_text.sub(r'', text)

    def remove_url(self, text):
        clean_text = re.compile('https?://\S+|www\.\S+')
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


    