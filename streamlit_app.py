import streamlit as st
from src.evaluate_model import load_model_and_tokenizer, predict_sentiments
from src.fetch_reviews import fetch_reviews

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer('trained_model_LSTM.h5', 'tokenizer.json')

def main():
    st.title("Movie Reviews Sentiment Analysis")
    
    movie_name = st.text_input("Enter the name of the movie:")
    
    if st.button("Analyze"):
        if movie_name:
            reviews = fetch_reviews(movie_name)
            if not reviews:
                st.error("No reviews found for this movie.")
            else:
                sentiments, predictions_prob = predict_sentiments(reviews, model, tokenizer)
                positive_reviews = sum(sentiments)
                total_reviews = len(sentiments)
                negative_reviews = total_reviews - positive_reviews
                
                st.write(f"Total Reviews: {total_reviews}")
                st.write(f"Positive Reviews: {positive_reviews}")
                st.write(f"Negative Reviews: {negative_reviews}")
                
                # for review, sentiment, prob in zip(reviews, sentiments, predictions_prob):
                #     st.write(f"Review: {review}")
                #     st.write(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}, Probability: {prob[0]}")
        else:
            st.error("Please enter a movie name.")
            
if __name__ == "__main__":
    main()