from src.download_data import download_and_save_data
from src.data_preprocessing import load_and_preprocess_data, get_word_index, decode_review
from src.rnn_model import build_rnn_model, train_rnn_model
from src.evaluate_model import evaluate_model, plot_roc

download_and_save_data('data')

train_data, train_labels, test_data, test_labels, tokenizer = load_and_preprocess_data('data')

word_index = get_word_index(tokenizer)

model_type = 'LSTM'  # Change this to 'SimpleRNN' or 'GRU' to use different models
model = build_rnn_model(train_data.shape, vocab_size=10000, model_type=model_type)
model = train_rnn_model(model, train_data, train_labels, epochs=10, batch_size=64)

accuracy, precision, recall, f1, predictions_prob = evaluate_model(model, test_data, test_labels)

print(f"Model Type: {model_type}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

plot_roc(test_labels, predictions_prob, model_type)