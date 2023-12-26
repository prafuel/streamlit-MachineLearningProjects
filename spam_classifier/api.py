
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


def text_transform(text : str) :
    punctuation = string.punctuation
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [token for token in tokens if token.isalnum()]
    tokens = [ps.stem(token) for token in tokens if token not in stopwords.words('english') and token not in punctuation]
    # tokens = [ps.stem(token) for token in tokens]
    
    return " ".join(tokens)


import pickle
cv = pickle.load(open("./cv.pkl", "rb"))
mnb = pickle.load(open("./mnb.pkl", "rb"))

user_input = "hi how are you"

def is_spam(user_input) :
    # 1. preprocess
    text = text_transform(user_input)

    # 2. vectorizer
    vector = cv.transform([text])

    # 3. predict
    predicted = mnb.predict(vector)[0]

    # 4. display
    if predicted == 0:
        return 0
    return 1

# frontend
import streamlit as st

st.title("Message spam classifier")
user_input = st.text_area("Enter the Message to Verify")

if st.button("Predict"):
    result = is_spam(user_input)
    if result == 0:
        st.text("Not Spam")
    else:
        st.text("Spam")

