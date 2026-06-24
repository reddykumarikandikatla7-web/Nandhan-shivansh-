import streamlit as st
import pickle

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam Email Classifier")

# User input
message = st.text_area("Enter message")

# Button
if st.button("Check"):
    message_vector = vectorizer.transform([message])
    result = model.predict(message_vector)

    if result[0] == "spam":
        st.error("This is Spam")
    else:
        st.success("This is Normal")