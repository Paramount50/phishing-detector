import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


model = joblib.load("phishing_detector.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

st.set_page_config(page_title="Phishing Email Detector", page_icon="🤓👆")
st.title("Phishing Email Detector 🤓👆")
st.write("Paste an email below and check if it's Phishing or Legit!")

user_input = st.text_area("Email Text 👀:", height=200)

if st.button("Check Email 🤔"):
    if user_input.strip() == "":
        st.warning("Enter some text 😡")
    else:
        clean = preprocess(user_input)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        if prediction == 1:
            st.error(f"😲 This email is PHISHING! (Confidence: {proba[1]*100:.2f}%)")
        else:
            st.success(f"😃 This email looks LEGIT. (Confidence: {proba[0]*100:.2f}%)")
