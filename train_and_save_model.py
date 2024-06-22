import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.data_preprocessing import load_and_preprocess_data
import json

train_data, train_labels, test_data, test_labels, tokenizer = load_and_preprocess_data('data', maxlen=500, num_words=10000)

def build_rnn_model(input_shape, vocab_size, model_type='SimpleRNN'):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_shape[1]))
    
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(32))
    elif model_type == 'LSTM':
        model.add(LSTM(32))
    elif model_type == 'GRU':
        model.add(GRU(32))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_type = 'LSTM'  # Choose 'SimpleRNN', 'LSTM', or 'GRU'
model = build_rnn_model(train_data.shape, vocab_size=10000, model_type=model_type)
model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

model.save(f'trained_model_{model_type}.h5')

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f)