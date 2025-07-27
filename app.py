import streamlit as st
import pickle
import re


# Import NLTK for stemming and stopwords
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    nltk.download('stopwords', quiet=True)
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except:
    stop_words = set()
    NLTK_AVAILABLE = False

@st.cache_resource
def load_models():
    try:
        with open('trained_model.sav', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_models()

# Your original negative keywords
negative_keywords = [
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "nor",
    "hardly", "barely", "scarcely", "without", "lack", "isn't", "aren't", "wasn't",
    "weren't", "won't", "wouldn't", "shouldn't", "couldn't", "don't", "doesn't",
    "didn't", "can't", "cannot", "ain't",
    "bad", "worst", "awful", "terrible", "horrible", "disgusting", "dreadful",
    "lousy", "pathetic", "useless", "annoying", "boring", "disappointed",
    "disappointing", "ridiculous", "unacceptable", "unbearable", "mediocre",
    "regret", "hate", "hated", "sucks", "poor", "weak",
    "problem", "issue", "complaint", "flawed", "difficult", "confusing", "unclear",
    "slow", "noisy", "incomplete", "buggy", "inconsistent", "unreliable", "glitch",
    "error", "fail", "failed", "lacking", "missing"
]

def classify_neutral_review(text):
    text_lower = text.lower()
    for word in negative_keywords:
        if word in text_lower:
            return 'Negative'
    return 'Positive'

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    if NLTK_AVAILABLE:
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    else:
        stemmed_content = [word for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_sentiment(comment):
    if not comment.strip():
        return "Please enter a comment", ""
    
    if model is None or vectorizer is None:
        return "Error: Models not loaded", "Check model files."
    
    # Rule-based override (your original logic)
    if classify_neutral_review(comment) == "Negative":
        return "Negative", "Rule-based detection"
    
    # ML model
    stemmed = stemming(comment)
    if not stemmed:
        return "Neutral", "No meaningful content"
    
    try:
        vector = vectorizer.transform([stemmed])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0].max()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment, f"ML model (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error: {str(e)}", "Model prediction failed"

# Streamlit UI
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Comment Sentiment Analysis")

if model is None or vectorizer is None:
    st.error("Models not loaded. Ensure 'trained_model.sav' and 'vectorizer.pkl' are in the directory.")
    st.stop()

user_input = st.text_area("💬 Enter a YouTube comment:", height=100)

if st.button("🔍 Analyze Sentiment"):
    result, explanation = predict_sentiment(user_input)
    
    if result == "Positive":
        st.success(f"✅ Sentiment: {result}")
    elif result == "Negative":
        st.error(f"❌ Sentiment: {result}")
    elif "Neutral" in result:
        st.warning(f"⚠️ Sentiment: {result}")
    else:
        st.info(f"ℹ️ {result}")
    
    if explanation:
        st.info(f"🔍 Method: {explanation}")