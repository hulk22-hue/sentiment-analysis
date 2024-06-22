from flask import Flask, request, render_template
from src.data_preprocessing import preprocess_text
from src.evaluate_model import load_model_and_tokenizer, predict_sentiments
from src.fetch_reviews import fetch_reviews
import os

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer('trained_model_LSTM.h5', 'tokenizer.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    movie_name = request.form['movie_name']
    reviews = fetch_reviews(movie_name)
    
    if not reviews:
        return render_template('index.html', message="No reviews found for this movie.")
    
    sentiments, predictions_prob = predict_sentiments(reviews, model, tokenizer)
    
    for review, sentiment, prob in zip(reviews, sentiments, predictions_prob):
        print(f"Review: {review}")
        print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}, Probability: {prob[0]}")

    positive_reviews = sum(sentiments)
    total_reviews = len(sentiments)
    negative_reviews = total_reviews - positive_reviews

    return render_template('index.html', movie_name=movie_name, total_reviews=total_reviews,
                           positive_reviews=positive_reviews, negative_reviews=negative_reviews)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')