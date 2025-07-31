# Phishing Email Detector 🤓👆

A machine learning project to classify emails as **Phishing** or **Legit** using Naive Bayes and TF-IDF vectorization.  
Includes a Streamlit app for interactive use.

## Features
- Preprocessing of email text
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Streamlit web interface
- Confidence scores for predictions

## 📂 Project Structure
phishing-detector/
│
├── data/ # dataset (not uploaded by default)
├── src/ # training code
├── app.py # Streamlit app
├── phishing_detector.pkl # trained model
├── vectorizer.pkl # vectorizer
├── requirements.txt # dependencies
└── README.md


## Running Locally
pip install -r requirements.txt
streamlit run app.py


## To retrain the model:
python src/train_model.py

