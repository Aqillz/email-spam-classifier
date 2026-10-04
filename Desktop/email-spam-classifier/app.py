import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Setup the web page
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")
st.write("Type a message below to find out if it is spam or genuine.")

# 2. Train and cache the model
@st.cache_resource
def train_model():
    # Dynamically get the exact folder path where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'spam.csv')
    
    # Load data using the new dynamic path
    df = pd.read_csv(data_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_tfidf = vectorizer.fit_transform(df['message'])
    
    # Train Naive Bayes
    model = MultinomialNB()
    model.fit(X_tfidf, df['label'])
    
    return vectorizer, model

# Show a loading spinner during the initial training phase
with st.spinner("Training the AI model..."):
    vectorizer, model = train_model()

# 3. Create the user interface
user_input = st.text_area("Enter an email or text message:", height=150)

if st.button("Classify Message"):
    if user_input.strip():
        # Convert the user's text into numbers using our trained vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # Make a prediction
        prediction = model.predict(input_tfidf)[0]
        
        # Display the result
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED** - This message looks suspicious.")
        else:
            st.success("✅ **NOT SPAM (HAM)** - This message looks safe.")
    else:
        st.warning("Please enter a message to classify.")