import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb

def download_and_save_data(file_path):
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    
    # Decode the reviews
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = {value: key for key, value in word_index.items()}
    train_reviews = [" ".join([reverse_word_index.get(i, '?') for i in review]) for review in train_data]
    test_reviews = [" ".join([reverse_word_index.get(i, '?') for i in review]) for review in test_data]

    train_df = pd.DataFrame({'review': train_reviews, 'label': train_labels})
    test_df = pd.DataFrame({'review': test_reviews, 'label': test_labels})

    train_df.to_csv(os.path.join(file_path, 'train_reviews.csv'), index=False)
    test_df.to_csv(os.path.join(file_path, 'test_reviews.csv'), index=False)
    print(f"IMDb dataset downloaded and saved to {file_path}")
