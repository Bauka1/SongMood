from openai import OpenAI
import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from googleapiclient.discovery import build
import re

# --- 🔑 API ключи ---
OPENAI_API_KEY = ""
YOUTUBE_API_KEY = ""

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

# --- 🔙 Резервные песни по эмоциям ---
fallback_songs = {
    "joy": ["Pharrell Williams - Happy", "Katy Perry - Roar", "Bruno Mars - Uptown Funk"],
    "sadness": ["Adele - Someone Like You", "Lewis Capaldi - Someone You Loved", "Billie Eilish - When The Party’s Over"],
    "anger": ["Linkin Park - Numb", "Eminem - Lose Yourself", "Rage Against The Machine - Killing In The Name"],
    "fear": ["Imagine Dragons - Demons", "Billie Eilish - Bury A Friend", "Radiohead - Creep"],
    "love": ["Ed Sheeran - Perfect", "John Legend - All Of Me", "Elvis Presley - Can’t Help Falling In Love"],
    "surprise": ["Coldplay - Viva La Vida", "Queen - Bohemian Rhapsody", "BTS - Dynamite"]
}

# --- Очистка текста от смайлов ---
def clean_text(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # смайлы
        u"\U0001F300-\U0001F5FF"  # символы
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# --- Поиск песни на YouTube ---
def search_youtube(song_name):
    try:
        request = youtube.search().list(
            q=f"{song_name} official music video",
            part="snippet",
            type="video",
            videoCategoryId="10",  # категория Music
            maxResults=1,
            order="relevance"
        )
        response = request.execute()
        if response["items"]:
            item = response["items"][0]
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            # Исключаем Shorts
            if "shorts" not in url.lower() and "shorts" not in title.lower():
                return {"title": title, "thumbnail": thumbnail, "url": url}
    except Exception as e:
        print("YouTube API error:", e)
    return None

# --- Генерация песен GPT ---
def suggest_song_list(emotion, text):
    prompt = f"""
    Назови 5 популярных песен, которые лучше всего подходят под эмоцию "{emotion}".
    Ответь строго в формате списка:
    1. Исполнитель - Название
    2. Исполнитель - Название
    ...
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        songs = re.findall(r"\d+\.\s*(.+)", content)
        return songs if songs else None
    except Exception as e:
        print("OpenAI API error:", e)
        return None

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="🎶 SongMood AI", layout="centered")
st.title("🎧 SongMood — Музыка по вашему настроению")

st.write("Введите описание вашего состояния (на любом языке), и приложение подберёт песни 🎵")

text_input = st.text_area("✍️ Опишите своё настроение:")

if st.button("🎧 Подобрать песню"):
    if not text_input.strip():
        st.warning("⚠️ Пожалуйста, введите текст.")
    else:
        # 🧹 Очистка от смайлов
        cleaned_text = clean_text(text_input)

        # 1️⃣ Перевод текста
        translated = GoogleTranslator(source='auto', target='en').translate(cleaned_text)
        st.info(f"🔄 Перевод текста: *{translated}*")

        # 2️⃣ Анализ эмоций
        predictions = emotion_classifier(translated)[0]
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        main_emotion = sorted_preds[0]['label']
        emoji = emotion_emojis.get(main_emotion, "🙂")

        st.markdown(f"### {emoji} Определённая эмоция: **{main_emotion.capitalize()}**")

        # 3️⃣ Получаем список песен
        songs = suggest_song_list(main_emotion, translated)

        if not songs:
            st.warning("❌ Не удалось получить список песен от GPT. Используем резервный список.")
            songs = fallback_songs.get(main_emotion, ["Imagine Dragons - Believer"])

        # 4️⃣ Поиск на YouTube
        st.subheader("🎵 Рекомендованные треки:")
        for song in songs:
            song_info = search_youtube(song)
            if song_info:
                st.markdown(f"**{emoji} {song_info['title']}**")
                st.image(song_info['thumbnail'], width=320)
                st.markdown(f"[▶️ Слушать на YouTube]({song_info['url']})")
                st.markdown("---")
            else:
                st.warning(f"Не удалось найти видео: {song}")
