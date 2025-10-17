from openai import OpenAI
import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from googleapiclient.discovery import build

# --- 🔑 API ключи ---
OPENAI_API_KEY = "sk-proj-uzlloZ4aJ0y2tCskJu9uj2d7JQX8Hf7ieDK0HiFB2pHXQgtH6krarWAh_JsxAp4CItzwUzXbOGT3BlbkFJR9bEC-2ZXXiIni0UoIsVCWPvq9ZT9oYitzinuHR5U45qu6Dcng3h0I_H5i9W2NpXvQFWN4SkEAQ"
YOUTUBE_API_KEY = "AIzaSyB6RlfddksDOhIX95ah7PKIOkcEf81XUbc"

client = OpenAI(api_key=OPENAI_API_KEY)
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# --- Модель анализа эмоций ---
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# --- Сопоставление эмоций со смайликами ---
emotion_emojis = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# --- Поиск песни на YouTube ---
def search_youtube(song_name):
    try:
        request = youtube.search().list(
            q=song_name,
            part="snippet",
            type="video",
            maxResults=1
        )
        response = request.execute()
        if response["items"]:
            item = response["items"][0]
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            return {"title": title, "thumbnail": thumbnail, "url": url}
    except Exception as e:
        print("YouTube API error:", e)
    return None

# --- Генерация песни с помощью OpenAI ---
def suggest_song(emotion, text):
    prompt = f"""
    Найди популярную песню, подходящую под настроение.
    Эмоция: {emotion}.
    Описание: "{text}".
    Ответь только в формате "Исполнитель - Название".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API error:", e)
        return "Не удалось подобрать песню 😔"

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="🎶 SongMood AI", layout="centered")
st.title("🎧 SongMood — Музыка по вашему настроению")

st.write("Введите описание вашего состояния (на любом языке), и приложение подберёт песни 🎵")

text_input = st.text_area("✍️ Опишите своё настроение:")

if st.button("🎧 Подобрать песню"):
    if not text_input.strip():
        st.warning("⚠️ Пожалуйста, введите текст.")
    else:
        # 1️⃣ Перевод текста
        translated = GoogleTranslator(source='auto', target='en').translate(text_input)
        st.info(f"🔄 Перевод текста: *{translated}*")

        # 2️⃣ Анализ эмоций
        predictions = emotion_classifier(translated)[0]
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        main_emotion = sorted_preds[0]['label']
        emoji = emotion_emojis.get(main_emotion, "🙂")

        # 3️⃣ Вывод эмоции (визуально)
        st.markdown(f"### {emoji} Определённая эмоция: **{main_emotion.capitalize()}**")

        # 4️⃣ Берём топ-2 эмоции, если их несколько
        strong_emotions = [p for p in sorted_preds if p['score'] > 0.2][:2]

        # 5️⃣ Подбор песен
        st.subheader("🎵 Рекомендованные треки:")
        for e in strong_emotions:
            emotion_name = e['label']
            song = suggest_song(emotion_name, translated)
            song_info = search_youtube(song)

            if song_info:
                st.markdown(f"**{emotion_emojis.get(emotion_name, '🎶')} {emotion_name.capitalize()} — {song_info['title']}**")
                st.image(song_info['thumbnail'], width=320)
                st.markdown(f"[▶️ Слушать на YouTube]({song_info['url']})")
                st.markdown("---")
            else:
                st.warning(f"Не удалось найти видео для {emotion_name}")
