# 🎬 YouTube Sentiment Analysis with Enhanced Detection

This project is a web-based YouTube comment sentiment analyzer built using **Streamlit**, **scikit-learn**, and **NLTK**. It uses a combination of **machine learning** and **rule-based logic** to classify comments as **Positive**, **Negative**, or **Neutral/Mixed** — while reducing false positives commonly found in simple sentiment models.

---

## 🚀 Live Demo
Try it out (if deployed): [https://youtube-sentiment-analysis-cmznc5leg8bd9vdzpiegdz.streamlit.app/](https://youtube-sentiment-analysis-cmznc5leg8bd9vdzpiegdz.streamlit.app/)

---

## ✨ Features

- ✅ **Hybrid Sentiment Detection** (ML + Rule-Based)
- 🔍 Detects:
  - Negated positives (e.g. "not good")
  - Conditional negatives (e.g. "good but slow")
  - Sarcasm-like patterns (e.g. "perfect... not")
  - Low ratings (e.g. "1/10", "1 star")
- 💬 Predefined test examples for quick evaluation
- 📊 Confidence-based classification with fallback logic
- 📦 Model and vectorizer caching for faster loads
- 🔧 Debug mode with preprocessing insights

---

## 🧠 Model Details

- **Classifier**: Logistic Regression (or any scikit-learn model)
- **Vectorizer**: TF-IDF
- **Text Preprocessing**:
  - Lowercasing
  - Stopword removal
  - Stemming using NLTK’s `PorterStemmer`

---


---

## 🛠️ Setup Instructions

### 1. 🔃 Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-sentiment-app.git
cd youtube-sentiment-app

```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app locally

```bash
streamlit run app.py
```

