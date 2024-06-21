import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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