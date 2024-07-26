import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from src.data_preprocessing import load_and_preprocess_data
import os

def create_model():
    """Load the BERT model for sequence classification."""
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, train_dataset, test_dataset, batch_size=8, epochs=3):
    """Train the BERT model."""
    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq='epoch')

    history = model.fit(train_dataset.shuffle(10000).batch(batch_size),
                        epochs=epochs,
                        validation_data=test_dataset.batch(batch_size),
                        callbacks=[cp_callback])
    return history

def save_model_and_tokenizer(model, tokenizer, save_directory):
    """Save the trained model and tokenizer."""
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

def main(file_path, save_directory, max_length=128, batch_size=8, epochs=3):
    """Main function to load data, preprocess, train and save the BERT model."""

    train_dataset, test_dataset, tokenizer = load_and_preprocess_data(file_path, max_length)

    model = create_model()

    train_model(model, train_dataset, test_dataset, batch_size, epochs)

    save_model_and_tokenizer(model, tokenizer, save_directory)

if __name__ == "__main__":
    file_path = 'data'  
    save_directory = 'fine_tuned_bert'  
    main(file_path, save_directory)