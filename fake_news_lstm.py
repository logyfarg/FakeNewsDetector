import pandas as pd
import string
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download('stopwords')

# Load and combine data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/Real.csv")

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], axis=0).sample(frac=1).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean'] = df['title'].apply(clean_text)

# Tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean'])
sequences = tokenizer.texts_to_sequences(df['clean'])
padded = pad_sequences(sequences, maxlen=500)

X = padded
y = df['label'].values

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Embedding(5000, 64, input_length=500))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save("model/fake_news_model.h5")
print("✅ Model trained and saved.")

# Predict interactively
def predict_news(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=500)
    pred = model.predict(pad)[0][0]
    return "REAL ✅" if pred > 0.5 else "FAKE ❌"

while True:
    user_input = input("📰 Type a headline (or 'exit'): ")
    if user_input.lower() == 'exit':
        print("Byeee")
        break
    print("🔍", predict_news(user_input))
