# 📰 Fake News Detector (LSTM)

This project uses deep learning (LSTM) to detect whether a news headline is real or fake.  
It’s built with TensorFlow and trained on real-world news datasets.

## 💡 How it Works
- Cleans the text (removes links, punctuation, etc.)
- Uses a tokenizer to convert text into sequences
- Feeds the sequences into an LSTM model
- Predicts: "REAL ✅" or "FAKE ❌"

## 📁 Folder Structure
- `fake_news_lstm.py` – main Python code
- `data/` – small dataset (for testing)
- `model/` – trained model (optional)
- `requirements.txt` – dependencies

## 🚀 How to Run

```bash
pip install -r requirements.txt
python fake_news_lstm.py
📸 Sample Test
"NASA confirms water on Mars."
✅ REAL

"Aliens take over the Eiffel Tower."
❌ FAKE
Made with 💖 by Logina Mahmoud
