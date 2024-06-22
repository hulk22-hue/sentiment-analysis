import os
from src.download_data import download_and_save_data
from src.data_preprocessing import load_and_preprocess_data, preprocess_text, get_word_index
from src.evaluate_model import load_model_and_tokenizer, predict_sentiments, evaluate_model, plot_roc
from src.download_data import download_and_save_data

if not os.path.exists('images'):
    os.makedirs('images')

# download_and_save_data('data')

train_data, train_labels, test_data, test_labels, tokenizer = load_and_preprocess_data('data')

word_index = get_word_index(tokenizer)
reverse_word_index = {value: key for key, value in word_index.items()}
test_reviews = [' '.join([reverse_word_index.get(i, '?') for i in review]) for review in test_data]

# test_reviews = test_data.tolist()

model, tokenizer = load_model_and_tokenizer('trained_model_LSTM.h5', 'tokenizer.json')

_, predictions_prob = predict_sentiments(test_reviews, model, tokenizer)

accuracy, precision, recall, f1, predictions_prob = evaluate_model(model, test_data, test_labels)

print(f"Model Type: LSTM")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

plot_roc(test_labels, predictions_prob, "LSTM")