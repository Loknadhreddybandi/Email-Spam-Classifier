import streamlit as st
import pickle
import string
import nltk

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric characters
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app setup
st.title("Email/SMS Spam Classifier")

# User input for SMS or email
input_sms = st.text_area("Enter the message")

# When the Predict button is pressed
if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])

    # Predict using the trained model
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
