# src/data_preprocessing.py
import os
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

def load_and_preprocess_data(file_path, max_length=128):
    """Load and preprocess the IMDb dataset."""
    train_df = pd.read_csv(os.path.join(file_path, 'train_reviews.csv'))
    test_df = pd.read_csv(os.path.join(file_path, 'test_reviews.csv'))

    train_texts = train_df['review'].values
    train_labels = train_df['label'].values
    test_texts = test_df['review'].values
    test_labels = test_df['label'].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(
        list(train_texts), 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors='tf'
    )

    test_encodings = tokenizer(
        list(test_texts), 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors='tf'
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    ))

    return train_dataset, test_dataset, tokenizer

def preprocess_text(reviews, tokenizer, max_length=128):
    """Tokenize the reviews using BERT tokenizer."""
    encodings = tokenizer(
        list(reviews), 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors='tf'
    )
    return encodings

