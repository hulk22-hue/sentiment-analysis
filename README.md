# Sentiment Analysis on Movie Reviews

This project aims to perform sentiment analysis on movie reviews using a Recurrent Neural Network (RNN). The app is built using Streamlit and deployed on Streamlit Sharing. The application allows users to enter the name of a movie, fetches reviews from IMDb, processes them, and predicts whether the sentiment is positive or negative.

## Features

- Data preprocessing and feature extraction using TensorFlow/Keras.
- Sentiment analysis using a trained LSTM model.
- Fetching movie reviews from IMDb.
- Flask web application to interact with the sentiment analysis model.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Pip (Python package installer)
- Git
- Heroku CLI (for deployment)

### Install Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-app.git
    cd sentiment-analysis-app
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Optional: Download Data

If you want to download the IMDb dataset manually and save it as CSV files, you can use the `download_data.py` script.

1. Run the `download_data.py` script:

    ```bash
    python src/download_data.py
    ```

2. This will create `train_reviews.csv` and `test_reviews.csv` in the `data` directory.

### Running the App Locally

1. Ensure you have the trained model and tokenizer files (`trained_model_LSTM.h5` and `tokenizer.json`) in the project directory.

2. Run the Flask application:

    ```bash
    python app.py
    ```

3. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Deployment

### Deploying to Heroku

1. Login to Heroku:

    ```bash
    heroku login
    ```

2. Create a new Heroku app:

    ```bash
    heroku create your-app-name
    ```

3. Add your files to Git and commit:

    ```bash
    git add .
    git commit -m "Initial commit"
    ```

4. Push your code to Heroku:

    ```bash
    git push heroku main  # If your branch is named 'main'
    ```

5. Open your app in the browser:

    ```bash
    heroku open
    ```

## Usage

1. Open the web application in your browser.
2. Enter the name of a movie and submit the form.
3. The app will fetch reviews from IMDb, analyze their sentiments, and display the results.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.