import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from src.data_preprocessing import preprocess_text

import json

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_model(model_path)
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    return model, tokenizer

def predict_sentiments(reviews, model, tokenizer):
    if isinstance(reviews, np.ndarray):
        reviews = reviews.tolist()
    preprocessed_reviews = preprocess_text(reviews, tokenizer)
    predictions_prob = model.predict(preprocessed_reviews)
    predictions = (predictions_prob > 0.5).astype("int32")
    return predictions, predictions_prob

def evaluate_model(model, test_data, test_labels):
    predictions_prob = model.predict(test_data)
    predictions = (predictions_prob > 0.5).astype("int32")

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    return accuracy, precision, recall, f1, predictions_prob

def plot_roc(test_labels, predictions_prob, model_name):
    
    ensure_dir('../images')
    fpr, tpr, _ = roc_curve(test_labels, predictions_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'../images/{model_name}_roc_curve.png')
    plt.show()