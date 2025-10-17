# 🎶 SongMood — Музыка по вашему настроению (AI версия)

SongMood — это интерактивное AI-приложение на **Streamlit**, которое анализирует эмоции пользователя по введённому тексту и подбирает подходящие песни с **YouTube**.

---

## 🚀 Функционал

- Определение эмоции пользователя   
- Автоматический перевод текста (если введён не на английском)  
- Генерация списка песен с помощью **OpenAI API**  
- Поиск треков и обложек на **YouTube API**  
- Удобный интерфейс для отображения эмоции и треков  

---

## 🧠 Пример использования

**Пользователь вводит:**
> “Я рад, но немного устал.”

**Приложение:**
- Определяет эмоции: 😄 *Joy*, 😢 *Sadness*  
- Подбирает несколько песен под каждую эмоцию  
- Показывает обложку и ссылку на YouTube 🎧  

---

## 🧩 Установка

1️⃣ Клонируйте проект:
```bash
git clone https://github.com/Bauka1/SongMood.git
cd SongMood

2️⃣ Установите зависимости:
pip install -r requirements.txt

3️⃣ Укажите ключи API:
# --- 🔑 API ключи ---
OPENAI_API_KEY = "sk-proj-uzlloZ4aJ0y2tCskJu9uj2d7JQX8Hf7ieDK0HiFB2pHXQgtH6krarWAh_JsxAp4CItzwUzXbOGT3BlbkFJR9bEC-2ZXXiIni0UoIsVCWPvq9ZT9oYitzinuHR5U45qu6Dcng3h0I_H5i9W2NpXvQFWN4SkEAQ"
YOUTUBE_API_KEY = "AIzaSyB6RlfddksDOhIX95ah7PKIOkcEf81XUbc"

4️⃣ Запустите приложение:
streamlit run app.py

🖼️ Пример интерфейса
<img width="955" height="626" alt="image" src="https://github.com/user-attachments/assets/eb649bb4-2142-4607-b207-afbdc382b143" />

