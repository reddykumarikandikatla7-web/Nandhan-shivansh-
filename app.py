import streamlit as st
import pickle

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam Email Classifier")

# Manual text input
message = st.text_area("Enter message")

# File upload option
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# Read uploaded file
if uploaded_file is not None:
    message = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Message:")
    st.write(message)

# Check button
if st.button("Check"):
    if message:
        message_vector = vectorizer.transform([message])
        result = model.predict(message_vector)

        if result[0] == "spam":
            st.error("This is Spam")
        else:
            st.success("This is Normal")
    else:
        st.warning("Please enter or upload a message")