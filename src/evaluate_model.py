# src/evaluate_model.py
import os
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load the fine-tuned BERT model and tokenizer."""
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def predict_sentiments(reviews, model, tokenizer):
    """Predict sentiments using the BERT model."""
    inputs = tokenizer(
        reviews,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    outputs = model(inputs)
    predictions = tf.nn.sigmoid(outputs.logits).numpy()
    sentiments = (predictions > 0.5).astype("int32")
    return sentiments, predictions


def evaluate_model(model, tokenizer, test_dataset, test_labels):
    """Evaluate the model on the test dataset."""
    test_reviews = [" ".join([str(token) for token in x[0]['input_ids'].numpy().tolist()]) for x in test_dataset]
    inputs = tokenizer(
        test_reviews,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    outputs = model(inputs)
    predictions = tf.nn.sigmoid(outputs.logits).numpy()
    sentiments = (predictions > 0.5).astype("int32")

    accuracy = accuracy_score(test_labels, sentiments)
    precision = precision_score(test_labels, sentiments)
    recall = recall_score(test_labels, sentiments)
    f1 = f1_score(test_labels, sentiments)

    return accuracy, precision, recall, f1, predictions

def plot_roc(test_labels, predictions_prob, model_name):
    """Plot the ROC curve."""
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
    plt.savefig(f'images/{model_name}_roc_curve.png')
    plt.show()

def main():
    file_path = 'data' 
    save_directory = 'fine_tuned_bert' 
    model, tokenizer = load_model_and_tokenizer(save_directory, save_directory)
    test_reviews = [
    "This movie was an absolute delight! The storyline was captivating and the characters were well-developed. I particularly enjoyed the performances of the lead actors who brought a lot of depth and emotion to their roles. The cinematography was stunning and the score complemented the film beautifully. It’s one of those movies that keeps you engaged from start to finish and leaves you with a warm feeling long after it’s over. Highly recommend!",
    "I found this movie to be incredibly disappointing. The plot was convoluted and the pacing was all over the place. The characters felt flat and uninteresting, and there was no chemistry between the leads. The dialogue was cringe-worthy at times and the special effects were subpar. It felt like a waste of time, and I wouldn’t recommend it to anyone. Save your money and watch something else.",
    "There were parts of this movie that I really enjoyed, and others that I found lacking. The initial setup was intriguing and had a lot of potential, but as the story progressed, it seemed to lose focus. Some scenes were beautifully shot and the acting was solid, but the script needed more work. It’s an okay film if you have nothing else to watch, but don’t go in with high expectations.",
    "What a fantastic film! The plot was well thought out and executed perfectly. The director did an excellent job of maintaining suspense and building tension throughout. The cast was phenomenal, with standout performances that added layers of complexity to the characters. The soundtrack was also a highlight, adding to the overall atmosphere of the movie. This is a must-watch for any movie lover!",
    "This was one of the worst movies I’ve seen in a long time. The story was predictable and the characters were one-dimensional. The acting was subpar, and the dialogue was unrealistic and forced. I found myself constantly checking the time, hoping it would end soon. It’s a shame because the premise had potential, but it was poorly executed. I would not recommend this movie to anyone."
]
    sentiments, predictions = predict_sentiments(test_reviews, model, tokenizer)
    print("Sentiments:", sentiments)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()