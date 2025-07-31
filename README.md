# Phishing Email Detector 🤓👆

A machine learning project to classify emails as **Phishing** or **Legit** using Naive Bayes and TF-IDF vectorization.  
Includes a Streamlit app for interactive use.

## 🚀 Features
- Preprocessing of email text
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Streamlit web interface
- Confidence scores for predictions

## 📂 Project Structure
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

## ▶️ Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔄 To Retrain the Model
```bash
python src/train_model.py
```

## 👨‍💻 Connect with Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ayushmaan-sinha-129b09277/)
