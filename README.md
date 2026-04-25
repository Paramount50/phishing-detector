# Phishing Email Detector 

This project is a machine learning tool built to classify emails as either Phishing or Legit.

It uses a Multinomial Naive Bayes classifier, which is trained on email text that has been processed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The project also includes an interactive web application built with Streamlit, allowing you to paste in an email and get an instant classification.

## Features
- Email Text Preprocessing: Cleans and prepares raw email text for analysis.
- TF-IDF Feature Extraction: Converts text into numerical features that the machine learning model can understand.
- Naive Bayes Classifier: Employs a Multinomial Naive Bayes model to perform the classification.
- Streamlit Web Interface: Provides a simple and interactive web app (run via app.py) to test the model in real-time.
- Confidence Scores: The application shows the model's confidence in its prediction, not just the final label.

## Project Structure
```
phishing-detector/
│
├── data/                     # dataset (not uploaded by default)
├── src/                      # training code
│   └── train_model.py
├── app.py                    # Streamlit app
├── phishing_detector.pkl      # trained model
├── vectorizer.pkl             # vectorizer
├── requirements.txt           # dependencies
└── README.md
```

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## To Retrain the Model
```bash
python src/train_model.py
```

## Connect with Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ayushmaan-sinha-129b09277/)
