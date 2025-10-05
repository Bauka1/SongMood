import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# Загружаем модель для анализа эмоций
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Словарь песен под эмоции
songs = {
    "joy": ["Happy - Pharrell Williams", "Good Life - OneRepublic"],
    "sadness": ["Someone Like You - Adele", "Fix You - Coldplay"],
    "anger": ["Breaking the Habit - Linkin Park", "Killing in the Name - RATM"],
    "fear": ["Creep - Radiohead", "Everybody Hurts - R.E.M."],
    "love": ["Perfect - Ed Sheeran", "All of Me - John Legend"],
    "surprise": ["Viva La Vida - Coldplay", "Firework - Katy Perry"]
}

# Словарь перевода русских эмоций
russian_to_english = {
    "радость": "joy",
    "грусть": "sadness",
    "печаль": "sadness",
    "злость": "anger",
    "страх": "fear",
    "любовь": "love",
    "удивление": "surprise"
}

# Streamlit UI
st.set_page_config(page_title="🎵 SongMood", layout="centered")
st.title("🎶 SongMood — Музыка по вашему настроению")

st.write("Введите описание настроения (русский или английский) или выберите эмоцию:")

# Ввод пользователя
user_input = st.text_input("Введите текст или эмоцию:")

if st.button("Подобрать музыку"):
    if user_input.strip():
        user_input_lower = user_input.strip().lower()

        # Проверка: ввёл ли пользователь эмоцию напрямую
        if user_input_lower in songs.keys():
            main_emotion = user_input_lower
            st.success(f"👉 Вы выбрали эмоцию: {main_emotion}")

        elif user_input_lower in russian_to_english.keys():
            main_emotion = russian_to_english[user_input_lower]
            st.success(f"👉 Вы выбрали эмоцию: {user_input_lower} ({main_emotion})")

        else:
            # Перевод текста
            translated_text = GoogleTranslator(source='auto', target='en').translate(user_input)
            st.info(f"🔄 Перевод текста: {translated_text}")

            # Анализ эмоций
            predictions = emotion_classifier(translated_text)[0]
            sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
            main_emotion = sorted_preds[0]['label']
            st.success(f"👉 Распознанная эмоция: {main_emotion}")

        # Подбор песен
        if main_emotion in songs:
            st.subheader("🎧 Рекомендованные песни:")
            for song in songs[main_emotion]:
                st.write(f"- {song}")
        else:
            st.error("😔 К сожалению, для этой эмоции пока нет песен в базе.")
    else:
        st.warning("⚠️ Пожалуйста, введите текст или эмоцию.")
