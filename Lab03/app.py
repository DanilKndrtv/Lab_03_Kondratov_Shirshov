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


# import streamlit as st
# import pandas as pd

# st.set_page_config(
#     page_title="Данные Титаника",
#     page_icon="🚢",
#     layout="wide"
# )

# st.title("🚢 Данные пассажиров Титаника")

# # Загрузка CSV файла пользователем
# uploaded_file = st.file_uploader("Выберите CSV файл с данными пассажиров", type="csv")

# if uploaded_file is not None:
#     df_csv = pd.read_csv(uploaded_file)
#     st.subheader("📊 Данные из вашего CSV файла")
    
#     # Выбор колонок для отображения
#     selected_columns = st.multiselect("Выберите колонки для отображения", df_csv.columns.tolist(), default=df_csv.columns.tolist())
#     st.dataframe(df_csv[selected_columns], use_container_width=True)

#     st.markdown("---")

#     # Фильтр по выжившим
#     survived_option = st.selectbox(
#         "Выберите категорию пассажиров",
#         ["Все", "Спасенные (1)", "Погибшие (0)"]
#     )

#     if survived_option == "Спасенные (1)":
#         df_filtered = df_csv[df_csv['Survived'] == 1]
#     elif survived_option == "Погибшие (0)":
#         df_filtered = df_csv[df_csv['Survived'] == 0]
#     else:
#         df_filtered = df_csv

#     st.subheader("📊 Отфильтрованные данные")
#     st.dataframe(df_filtered[selected_columns], use_container_width=True)

#     st.markdown("---")

#     # Графики
#     st.subheader("📈 Графическая статистика")

#     # Выбор категории для графика
#     chart_option = st.selectbox("Выберите, что визуализировать", ["Возраст по классам", "Количество пассажиров по полу", "Выживаемость по классу"])

#     if chart_option == "Возраст по классам":
#         age_data = df_filtered.groupby('Pclass')['Age'].mean().reset_index()
#         st.bar_chart(data=age_data, x='Pclass', y='Age')

#     elif chart_option == "Количество пассажиров по полу":
#         sex_data = df_filtered['Sex'].value_counts()
#         st.bar_chart(sex_data)

#     elif chart_option == "Выживаемость по классу":
#         survival_data = df_filtered.groupby('Pclass')['Survived'].mean().reset_index()
#         st.bar_chart(survival_data, x='Pclass', y='Survived')

# else:
#     st.info("Загрузите CSV файл для отображения данных.")















