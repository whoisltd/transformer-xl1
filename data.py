import re
import io
import time
import json
import nltk
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
class Dataset:
    def __init__(self, data_path, vocab_size = 10000, max_length = 120):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = 16
        self.trunc_type = 'post'
        self.oov_tok = '<OOV>'
        self.tokenizer_save = None

    def load_dataset(self):
        print("Loading dataset...")
        data = pd.read_csv(self.data_path)
        data = data.dropna()
        data['review'] = data['review'].str.lower()
        labels = self.encode_labels(data['sentiment'])
        sentences = list(self.clean_data(data['review']))
        self.save_clean_data(sentences, labels)
        sentences = self.tokenizer_data(sentences, self.vocab_size, self.max_length, self.trunc_type, self.oov_tok)
        self.save_tokenizer(self.tokenizer_save)
        print("Dataset loaded.")
        return sentences, labels

    def save_clean_data(self, sentences, labels):
        output = pd.DataFrame({'review': sentences,
                        'sentiment': labels})
        output.to_csv('submission.csv', index=False)
        print("Clean data saved.")

    def save_tokenizer(self, tokenizer):
        tokenizer_json = tokenizer.to_json()
        with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def build_dataset(self):
        sentences, labels = self.load_dataset()
        X_train, X_val, y_train, y_val = self.split_data(sentences, labels)
        print(type(X_train))
        return ((X_train, X_val), (y_train, y_val))
    
    def clean_data(self, texts):
        for i in tqdm(range(len(texts))):
            texts[i] = self.remove_html_tags(texts[i])
            texts[i] = self.remove_url(texts[i])
            texts[i] = self.remove_emoji(texts[i])
            texts[i] = self.remove_punctuation(texts[i])
            texts[i] = self.remove_stopwords(texts[i])
            texts[i] = self.lemma_traincorpus(texts[i])
        return np.array(texts)

    def split_data(self, sentences, labels, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=test_size, random_state=42)
        return X_train, X_val, y_train, y_val

    def encode_labels(self, data):
        return np.array([1 if label == 'positive' else 0 for label in data])

    def remove_html_tags(self, text):
        clean_text = re.compile(r'<.*?>')
        return clean_text.sub(r'', text)

    def remove_url(self, text):
        clean_text = re.compile(r'https?://\S+|www\.\S+')
        return clean_text.sub(r'', text)

    def remove_punctuation(self, text):
        exclude = string.punctuation
        for char in exclude:
            text = text.replace(char,'')
        return text
    
    def remove_emoji(self, text):
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
        stop_words = set(stopwords.words('english'))
        words = text.split()
        list_words = []
        for word in words:
            if word not in stop_words:
                list_words.append(word)
        return ' '.join(list_words)
    
    def lemma_traincorpus(self, data):
        lemmatizer=WordNetLemmatizer()
        out_data=[lemmatizer.lemmatize(word) for word in data]
        return out_data

    def tokenizer_data(self, sentences, vocab_size, max_length, trunc_type, oov_tok):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        # Generate the word index dictionary for the training sentences
        tokenizer.fit_on_texts(sentences)
        # word_index = tokenizer.word_index
        self.tokenizer_save = tokenizer
        # Generate and pad the training sequences
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
        return padded