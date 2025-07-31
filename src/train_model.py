
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


df = pd.read_csv("data/phishing_email.csv")
print(df.shape)
print(df.head())


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


stop_words = set(stopwords.words("english"))


def preprocess(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    tokens=[word for word in text.split() if word not in stop_words]
    return " ".join(tokens)


df['clean_text']=df['text_combined'].apply(preprocess)


vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df['clean_text'])
y=df['label']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=MultinomialNB()
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
print("accuracy",accuracy_score(y_test,y_pred))
print("classification report :\n",classification_report(y_test,y_pred))


def predict_email(text):
    clean=preprocess(text)
    vec=vectorizer.transform([clean])
    probas = model.predict_proba(vec)[0]
    prediction=model.predict(vec)[0]
    return {
        "prediction": "phishing" if prediction == 1 else "legit",
        "confidence": float(max(probas))
    }


joblib.dump(model,"phishing_detector.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Saved in:", os.getcwd())

