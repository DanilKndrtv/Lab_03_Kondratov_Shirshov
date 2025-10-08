import os
from dotenv import load_dotenv
import streamlit as st
import requests

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
st.write("HF_TOKEN:", HF_TOKEN)  # проверяем, что токен подхватился

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
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













