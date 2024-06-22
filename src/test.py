import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load the pre-trained model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_model(model_path)
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    return model, tokenizer

# Preprocess the reviews
def preprocess_text(reviews, tokenizer, maxlen=500):
    sequences = tokenizer.texts_to_sequences(reviews)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

# Predict sentiments
def predict_sentiments(reviews, model, tokenizer):
    preprocessed_reviews = preprocess_text(reviews, tokenizer)
    predictions_prob = model.predict(preprocessed_reviews)
    predictions = (predictions_prob > 0.5).astype("int32")
    return predictions, predictions_prob

# Test the model on known data
def test_model_on_known_data():
    model_path = 'trained_model_LSTM.h5'
    tokenizer_path = 'tokenizer.json'
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    # Print the model summary
    print(model.summary())

    # Example reviews for testing
    example_reviews = [
        "This movie was fantastic! The acting was great and the story was compelling.",
        "I didn't enjoy this movie. It was too long and the plot was boring.",
        "An average movie. Some good moments but overall not impressive.",
        "One of the best movies I've seen in a long time. Highly recommend it!",
        "Terrible movie. Waste of time."
    ]

    sentiments, predictions_prob = predict_sentiments(example_reviews, model, tokenizer)

    # Print the predictions for the example reviews
    for review, sentiment, prob in zip(example_reviews, sentiments, predictions_prob):
        print(f"Review: {review}")
        print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}, Probability: {prob[0]}")

if __name__ == "__main__":
    test_model_on_known_data()