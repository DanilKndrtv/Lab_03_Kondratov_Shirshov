# import os
# from dotenv import load_dotenv
# import streamlit as st
# import requests

# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# st.write("HF_TOKEN:", HF_TOKEN)  # проверяем, что токен подхватился

# API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
# headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     st.write("Status code:", response.status_code)
#     st.write("Text response:", response.text)  # для отладки
#     try:
#         return response.json()
#     except Exception as e:
#         return {"error": "Не удалось распарсить ответ", "raw_response": response.text}

# st.title("🤗 Hugging Face API + Streamlit")
# text = st.text_area("Введите текст:")
# if st.button("Отправить") and text:
#     output = query({"inputs": text})
#     st.write(output)

import streamlit as st
import pandas as pd
import numpy as np
import os

# Настройка страницы
st.set_page_config(
    page_title="Данные Титаника",
    page_icon="🚢",
    layout="wide"
)

# Заголовок
st.title("🚢 Данные пассажиров Титаника")

# === ДОБАВЛЕННАЯ СТРОКА ДЛЯ УКАЗАНИЯ ПУТИ К CSV ===
csv_path = "\Users\Данил\Downloads\titanic_train.csv"

if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    st.subheader("📊 Данные из вашего CSV файла")
    st.dataframe(df_csv, use_container_width=True)
    st.markdown("---")
else:
    st.info(f"Файл не найден по пути: {csv_path}")

st.markdown("""
Для просмотра данных только по спасенным или погибшим, выберите соответствующий пункт из списка:
""")

# Создаем выпадающий список для фильтрации
survived_option = st.selectbox(
    "**Значение поля Survived:**",
    ["Любое", "Спасенные (1)", "Погибшие (0)"],
    index=0
)

# Разделитель
st.markdown("---")

# Создаем данные для таблицы (на основе вашего примера)
data = {
    'Класс обслуживания': ['1 класс', '2 класс', '3 класс'],
    'Средний возраст': [38.2, 29.9, 25.1]
}

df = pd.DataFrame(data)

# Добавляем столбец с Survived в зависимости от выбора
if survived_option == "Спасенные (1)":
    df.insert(0, 'Survived', [1, 1, 1])
    st.subheader("📊 Данные по спасенным пассажирам")
elif survived_option == "Погибшие (0)":
    df.insert(0, 'Survived', [0, 0, 0])
    st.subheader("📊 Данные по погибшим пассажирам")
else:
    df.insert(0, 'Survived', [0, 1, 2])
    st.subheader("📊 Общие данные по пассажирам")

# Отображаем таблицу с стилями
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# Дополнительная стилизация для лучшего визуала
st.markdown("""
<style>
    .stDataFrame {
        font-size: 16px;
    }
    div[data-testid="stDataFrame"] table {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Информация под таблицей
st.markdown("---")
st.markdown("""
**Примечание:**
- **Survived = 0** - пассажир погиб
- **Survived = 1** - пассажир спасся
- Данные показывают средний возраст пассажиров по классам обслуживания
""")

# Дополнительные визуализации (опционально)
with st.expander("📈 Дополнительная статистика"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1 класс", "38.2 лет", "Старше")
    
    with col2:
        st.metric("2 класс", "29.9 лет", "Средний")
    
    with col3:
        st.metric("3 класс", "25.1 лет", "Младше") что этот код еще вычисляет





