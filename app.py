import streamlit as st
from transformers import pipeline

st.title("🤖 Hugging Face + Streamlit Demo")

# Загружаем модель
model = pipeline("sentiment-analysis")

# Интерфейс
user_input = st.text_input("Введите текст для анализа:")

if user_input:
    result = model(user_input)[0]
    st.write(f"**Текст:** {user_input}")
    st.write(f"**Результат:** {result['label']} ({result['score']:.2f})")
