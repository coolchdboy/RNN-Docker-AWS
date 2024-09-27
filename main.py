# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Configure Streamlit App
st.set_page_config(
    page_title="Sentiment Analysis Using Simple RNN",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="auto"
)

# Step 4: Streamlit App Interface
st.markdown('<h1 style="font-size: 40px;">Sentiment Analysis Using Simple RNN</h1>', unsafe_allow_html=True)
st.write('Enter a movie review to analyze it as positive or negative.')

# User input
user_input = st.text_area('Enter your movie review here:', height=150)

if st.button('Analyze'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
    else:
        st.write('Please enter a valid movie review.')
else:
    st.write('Please enter a movie review and click "Analyze".')
