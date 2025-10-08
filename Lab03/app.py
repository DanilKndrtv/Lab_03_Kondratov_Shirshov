import os
from dotenv import load_dotenv
import streamlit as st
import requests
import pandas as pd

# Загружаем токен
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"Ошибка: {response.status_code}", "raw_response": response.text}
    try:
        return response.json()
    except:
        return {"error": "Не удалось распарсить ответ", "raw_response": response.text}

st.set_page_config(page_title="Анализ эмоций", page_icon="🤗", layout="centered")
st.title("🤗 Анализ эмоций в тексте (Hugging Face)")

text = st.text_area("Введите текст для анализа:")

if st.button("Отправить") and text:
    with st.spinner("Анализируем..."):
        output = query({"inputs": text})

    if "error" in output:
        st.error(output["error"])
        st.write(output.get("raw_response", ""))
    else:
        # Получаем список эмоций
        emotions = output[0]
        df = pd.DataFrame(emotions)
        df["score"] = df["score"].apply(lambda x: round(x * 100, 2))
        df = df.sort_values(by="score", ascending=False)

        # Находим главную эмоцию
        main_emotion = df.iloc[0]["label"]
        main_score = df.iloc[0]["score"]

        # Эмодзи для эмоций
        emoji_map = {
            "joy": "😊",
            "anger": "😡",
            "sadness": "😢",
            "fear": "😨",
            "surprise": "😲",
            "love": "❤️",
            "disgust": "🤢",
            "neutral": "😐"
        }
        emoji = emoji_map.get(main_emotion, "🤔")

        st.subheader(f"🎭 Основная эмоция: **{main_emotion.capitalize()} {emoji} ({main_score}%)**")

        # Таблица с результатами
        st.write("📊 Подробный анализ эмоций:")
        st.dataframe(df, use_container_width=True)

        # Визуализация
        st.bar_chart(df.set_index("label")["score"])

