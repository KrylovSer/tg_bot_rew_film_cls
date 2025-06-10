import os
import time
import logging
import re
import string
import pickle
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters.command import Command

from model import LSTMBahdanauAttentionEmb

# ========== Настройка логирования ==========
logging.basicConfig(level=logging.INFO, filename="info_log.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ========== Загрузка nltk ==========
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ========== Глобальные переменные и устройство ==========
device = torch.device("cpu")

# ========== Загрузка словаря и модели ==========
with open("vocab_emb.pkl", "rb") as f:
    vocab_to_int = pickle.load(f)

model = LSTMBahdanauAttentionEmb()
model.load_state_dict(torch.load("lstm_model_emb.pth", map_location=device))
model.eval()

# ========== Функции предобработки текста ==========

def data_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub('<.*?>', '', text)  # Удаляем html-теги
    text = ''.join([c for c in text if c not in string.punctuation])
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def padding(review_int: list, seq_len: int) -> np.array:
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[-seq_len:]
        features[i, :] = np.array(new)
    return features

def preprocess_single_string(
    input_string: str,
    seq_len: int,
    vocab_to_int: dict
) -> torch.Tensor:
    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        if word in vocab_to_int:
            result_list.append(vocab_to_int[word])
    result_padded = padding([result_list], seq_len)[0]
    return torch.LongTensor(result_padded).unsqueeze(0)

def classify(text: str):
    inp = preprocess_single_string(text, 135, vocab_to_int).to(device)
    with torch.inference_mode():
        output, _ = model(inp)
        prob = torch.sigmoid(output).squeeze().item()
        label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob

# ========== Инициализация бота и диспетчера ==========

API_TOKEN = os.getenv("BOT_TOKEN")
if API_TOKEN is None:
    logging.error("BOT_TOKEN не найден в переменных окружения!")
    exit(1)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# ========== Обработчики команд и сообщений ==========

@dp.message(Command(commands=['start', 'help']))
async def send_welcome(message: Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    logging.info(f"{user_name}:{user_id} - запустил бота.")
    await message.reply("Привет! Отправь отзыв о фильме на английском, и я скажу, позитивный он или нет.")

@dp.message()
async def handle_message(message: Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    text = message.text

    logging.info(f"{user_name}:{user_id} - сообщение: {text}")

    label, prob = classify(text)
    response = f"Класс: {label}\nВероятность(Positive): {prob:.2f}"
    await message.reply(response)

# ========== Запуск бота ==========

if __name__ == '__main__':
    logging.info("Бот запущен...")
    dp.run_polling(bot)
