import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.title("🤗 Hugging Face API + Streamlit")

text = st.text_area("Введите текст:")
if st.button("Отправить") and text:
    output = query({"inputs": text})
    st.write(output)
# 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



