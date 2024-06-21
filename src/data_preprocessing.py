import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def load_and_preprocess_data(file_path, maxlen=500, num_words=10000):
    train_df = pd.read_csv(os.path.join(file_path, 'train_reviews.csv'))
    test_df = pd.read_csv(os.path.join(file_path, 'test_reviews.csv'))

    train_reviews = train_df['review'].values
    train_labels = train_df['label'].values
    test_reviews = test_df['review'].values
    test_labels = test_df['label'].values

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_reviews)
    
    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    
    train_data = pad_sequences(train_sequences, maxlen=maxlen)
    test_data = pad_sequences(test_sequences, maxlen=maxlen)

    return train_data, train_labels, test_data, test_labels, tokenizer

def get_word_index(tokenizer):
    return tokenizer.word_index

def decode_review(text, tokenizer):
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
    return ' '.join([reverse_word_index.get(i, '?') for i in text])