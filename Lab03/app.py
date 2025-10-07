import os
from dotenv import load_dotenv
import streamlit as st
import requests

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
st.write("HF_TOKEN:", HF_TOKEN)  # проверяем, что токен подхватился

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.write("Status code:", response.status_code)
    st.write("Text response:", response.text)  # для отладки
    try:
        return response.json()
    except Exception as e:
        return {"error": "Не удалось распарсить ответ", "raw_response": response.text}

st.title("🤗 Hugging Face API + Streamlit")
text = st.text_area("Введите текст:")
if st.button("Отправить") and text:
    output = query({"inputs": text})
    st.write(output)

