import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the model
logreg = joblib.load('logistic_regression_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
process_reviews = {'reviews': ["example review for fitting"]}  # Dummy data for fitting
tfidf_vectorizer.fit(process_reviews['reviews'])

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['positive', 'negative'])  # Replace with the labels used during training

# Function to predict sentiment
def predict_sentiment(review):
    review_transformed = tfidf_vectorizer.transform([review])
    sentiment = logreg.predict(review_transformed)
    sentiment_label = label_encoder.inverse_transform(sentiment)
    return sentiment_label[0]

# Streamlit app
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of a user review.")

# Input from user
user_review = st.text_area("Enter your review:", height=150)

if st.button("Predict Sentiment"):
    if user_review.strip():
        result = predict_sentiment(user_review)
        st.write(f"**The sentiment of the review is:** {result}")
    else:
        st.warning("Please enter a review to predict the sentiment.")
