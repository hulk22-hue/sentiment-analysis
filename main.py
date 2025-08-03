import os
import tensorflow as tf
from src.data_preprocessing import load_and_preprocess_data, preprocess_text
from src.evaluate_model import load_model_and_tokenizer, predict_sentiments, evaluate_model, plot_roc
from src.download_data import download_and_save_data

if not os.path.exists('images'):
    os.makedirs('images')

# download_and_save_data('data')

file_path = 'data'
train_dataset, test_dataset, tokenizer = load_and_preprocess_data(file_path)

test_reviews = [" ".join([str(token) for token in x[0]['input_ids'].numpy().tolist()]) for x in test_dataset]
test_labels = [x[1].numpy() for x in test_dataset]

model_path = 'fine_tuned_bert'
model, tokenizer = load_model_and_tokenizer(model_path, model_path)

# _, predictions_prob = predict_sentiments(test_reviews, model, tokenizer)

batch_size = 16  
num_batches = len(test_reviews) // batch_size + 1

all_predictions = []
all_predictions_prob = []

for i in range(num_batches):
    batch_reviews = test_reviews[i * batch_size:(i + 1) * batch_size]
    if not batch_reviews:
        continue
    sentiments, predictions_prob = predict_sentiments(batch_reviews, model, tokenizer)
    all_predictions.extend(sentiments)
    all_predictions_prob.extend(predictions_prob)

all_predictions_prob = tf.convert_to_tensor(all_predictions_prob)
all_predictions_prob = all_predictions_prob.numpy()

accuracy, precision, recall, f1, _ = evaluate_model(model, tokenizer, test_dataset, test_labels)

print(f"Model Type: BERT")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

plot_roc(test_labels, predictions_prob, "BERT")