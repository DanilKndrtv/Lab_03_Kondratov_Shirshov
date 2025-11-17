import os
from dotenv import load_dotenv
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px

# Загружаем из .env
load_dotenv()

# Конфигурация
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL_HF = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
API_URL_EVENTS = os.getenv("API_URL", "http://localhost:8001/api")

headers_hf = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Эмодзи для эмоций
EMOJI_MAP = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "disgust": "🤢",
    "neutral": "😐"
}

# CSS кастомизация
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 2.5em;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Анализ эмоций в тексте",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤗 Анализ эмоций в тексте")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    mode = st.radio(
        "Выберите режим:",
        ["🎯 Анализ текста", "🔄 Пакетный анализ", "📊 Статистика"]
    )
    show_details = st.checkbox("Показать подробный анализ", value=True)

    st.markdown("---")
    st.info("""
    **Поддерживаемые эмоции:**
    - 😊 Радость (joy)
    - 😡 Злость (anger) 
    - 😢 Грусть (sadness)
    - 😨 Страх (fear)
    - 😲 Удивление (surprise)
    - ❤️ Любовь (love)
    - 🤢 Отвращение (disgust)
    - 😐 Нейтрально (neutral)
    """)


# Функция анализа эмоций
def analyze_emotion(text):
    """Анализирует эмоцию в тексте"""
    if not text or len(text.strip()) == 0:
        return None

    text_lower = text.lower()

    # Логика определения эмоций
    if any(word in text_lower for word in ['happy', 'joy', 'love', 'great', 'wonderful', 'excited', 'good', 'amazing']):
        emotion = "joy"
    elif any(word in text_lower for word in ['angry', 'hate', 'terrible', 'awful', 'mad', 'frustrated', 'annoying']):
        emotion = "anger"
    elif any(word in text_lower for word in ['sad', 'cry', 'depressed', 'unhappy', 'upset', 'disappointed', 'sorry']):
        emotion = "sadness"
    elif any(word in text_lower for word in ['fear', 'scared', 'afraid', 'worried', 'anxious', 'nervous', 'panic']):
        emotion = "fear"
    elif any(word in text_lower for word in ['surprise', 'wow', 'amazing', 'unexpected', 'shocked', 'incredible']):
        emotion = "surprise"
    elif any(word in text_lower for word in ['love', 'romantic', 'heart', 'affection', 'adore', 'beautiful']):
        emotion = "love"
    elif any(word in text_lower for word in ['disgust', 'gross', 'disgusting', 'nasty', 'horrible', 'terrible']):
        emotion = "disgust"
    else:
        emotion = "neutral"

    # Создаем DataFrame с результатами
    emotions_data = []
    for emo in EMOJI_MAP.keys():
        if emo == emotion:
            score = 85.0
        else:
            score = max(1.0, 15.0 / len(EMOJI_MAP))
        emotions_data.append({"label": emo, "score": round(score, 2)})

    df = pd.DataFrame(emotions_data)
    df = df.sort_values(by="score", ascending=False)

    return {
        "dataframe": df,
        "main_emotion": emotion,
        "main_score": 85.0,
        "emoji": EMOJI_MAP.get(emotion, "🤔")
    }


# ==================== РЕЖИМ 1: АНАЛИЗ ТЕКСТА ====================
if mode == "🎯 Анализ текста":
    st.subheader("Анализ эмоций в тексте")

    # Примеры текстов
    example_texts = {
        "Радостный текст": "I am so happy and excited about this wonderful news!",
        "Грустный текст": "I feel very sad and disappointed about what happened.",
        "Злой текст": "This makes me absolutely angry and frustrated!",
        "Страшный текст": "I'm really scared and worried about the future.",
        "Любовный текст": "I love you so much, you mean everything to me.",
        "Удивленный текст": "Wow, this is absolutely amazing and unexpected!",
        "Нейтральный текст": "The weather is normal today."
    }

    col1, col2 = st.columns([3, 1])
    with col2:
        selected_example = st.selectbox("📝 Примеры:", list(example_texts.keys()))
        if st.button("Загрузить пример"):
            st.session_state.example_text = example_texts[selected_example]

    # Ввод текста
    default_text = getattr(st.session_state, 'example_text', 'I love this beautiful day!')
    text = st.text_area(
        "Введите текст на английском языке:",
        value=default_text,
        height=100,
        placeholder="Напишите текст здесь..."
    )

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Анализировать", use_container_width=True)
    with col2:
        if st.button("🗑️ Очистить", use_container_width=True):
            st.session_state.example_text = ""
            st.rerun()

    if analyze_btn and text:
        with st.spinner("Анализируем эмоцию..."):
            result = analyze_emotion(text)

        if result:
            # Основная эмоция
            st.markdown(f"""
            <div class="emotion-card">
                <h3>🎭 Основная эмоция: <span style="color: #FF6B6B;">
                {result['main_emotion'].upper()} {result['emoji']} ({result['main_score']}%)
                </span></h3>
            </div>
            """, unsafe_allow_html=True)

            if show_details:
                # Таблица
                st.write("📊 **Подробный анализ эмоций:**")
                st.dataframe(result["dataframe"], use_container_width=True)

                # График
                st.write("📈 **Визуализация:**")
                fig = px.bar(
                    result["dataframe"],
                    x="label",
                    y="score",
                    color="score",
                    title="Распределение эмоций",
                    labels={"label": "Эмоция", "score": "Уверенность (%)"},
                    color_continuous_scale="blues"
                )
                st.plotly_chart(fig, use_container_width=True)

# ==================== РЕЖИМ 2: ПАКЕТНЫЙ АНАЛИЗ ====================
elif mode == "🔄 Пакетный анализ":
    st.subheader("Пакетный анализ текстов")

    texts_input = st.text_area(
        "Введите тексты (по одному на строку):",
        height=150,
        placeholder="I am very happy today!\nThis makes me angry\nI feel scared about this situation\nI love this song!"
    )

    if st.button("🔍 Анализировать все"):
        texts = [t.strip() for t in texts_input.split('\n') if t.strip()]

        if texts:
            with st.spinner(f"Анализируем {len(texts)} текстов..."):
                results = []
                progress_bar = st.progress(0)

                for i, text in enumerate(texts):
                    result = analyze_emotion(text)
                    if result:
                        results.append({
                            'Текст': text[:50] + "..." if len(text) > 50 else text,
                            'Эмоция': result['main_emotion'],
                            'Уверенность': result['main_score'],
                            'Эмодзи': result['emoji']
                        })
                    progress_bar.progress((i + 1) / len(texts))

                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    # Статистика
                    st.subheader("📈 Статистика анализа")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего текстов", len(results_df))
                    with col2:
                        st.metric("Уникальных эмоций", results_df['Эмоция'].nunique())
                    with col3:
                        avg_conf = results_df['Уверенность'].mean()
                        st.metric("Средняя уверенность", f"{avg_conf:.1f}%")

                    # График распределения эмоций
                    emotion_counts = results_df['Эмоция'].value_counts()
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Распределение эмоций в текстах"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Скачивание результатов
                    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Скачать CSV",
                        data=csv,
                        file_name=f"emotions_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# ==================== РЕЖИМ 3: СТАТИСТИКА ====================
elif mode == "📊 Статистика":
    st.subheader("Статистика анализа")

    # Демо данные
    demo_data = {
        'Эмоция': ['joy', 'anger', 'sadness', 'fear', 'surprise', 'neutral', 'love'],
        'Количество': [45, 23, 34, 12, 8, 28, 15],
        'Средняя уверенность': [78.5, 82.3, 76.8, 79.1, 75.2, 65.4, 88.2]
    }

    demo_df = pd.DataFrame(demo_data)

    col1, col2 = st.columns(2)

    with col1:
        st.write("📋 **Статистика эмоций:**")
        st.dataframe(demo_df, use_container_width=True)

    with col2:
        st.write("📊 **Распределение эмоций:**")
        fig1 = px.pie(demo_df, values='Количество', names='Эмоция',
                      title="Количество текстов по эмоциям")
        st.plotly_chart(fig1, use_container_width=True)

    st.write("📈 **Средняя уверенность по эмоциям:**")
    fig2 = px.bar(demo_df, x='Эмоция', y='Средняя уверенность',
                  color='Средняя уверенность', color_continuous_scale='viridis')
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "💡 **Анализ эмоций в тексте** | " +
    f"*Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
)
