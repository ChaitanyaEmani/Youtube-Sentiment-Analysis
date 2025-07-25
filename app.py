import streamlit as st
import pickle
import re
import os

# Try to import NLTK components with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Initialize stemmer and stopwords
    port_stem = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback to basic English stopwords if NLTK data unavailable
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                     'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                     'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                     'further', 'then', 'once'}
    
    NLTK_AVAILABLE = True
except ImportError:
    st.warning("NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False
    stop_words = set()

# Load model and vectorizer with better error handling
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer with caching"""
    model_path = 'trained_model.sav'
    vectorizer_path = 'vectorizer.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please ensure the file exists in the same directory.")
        return None, None
    
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Vectorizer file '{vectorizer_path}' not found. Please ensure the file exists in the same directory.")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        return None, None

# Load models
model, vectorizer = load_models()

# Enhanced negative keywords with more subtle negative patterns
negative_keywords = {
    # Strong negations
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "nor",
    "hardly", "barely", "scarcely", "without", "lack", "lacking", "missing", "cannot",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't",
    "couldn't", "don't", "doesn't", "didn't", "can't", "ain't",
    
    # Direct negative words
    "bad", "worst", "awful", "terrible", "horrible", "dreadful", "lousy", "pathetic",
    "useless", "boring", "disappointing", "ridiculous", "unacceptable", "unbearable",
    "mediocre", "cheap", "fake", "weak", "pointless", "meaningless", "stupid",
    "dumb", "annoying", "frustrating", "worthless", "poor", "inferior", "subpar",
    "inadequate", "deficient", "deplorable", "atrocious", "abysmal", "disgusting",
    
    # Technical issues
    "problem", "issue", "bug", "error", "glitch", "broken", "fail", "failed",
    "crash", "lag", "freeze", "slow", "delayed", "stuck", "unresponsive",
    "malfunction", "defective", "faulty", "corrupted", "damaged",
    
    # Emotional negatives
    "hate", "angry", "mad", "upset", "frustrated", "annoyed", "disappointed",
    "regret", "waste", "trash", "garbage", "scam", "misleading",
    
    # Subtle negatives that are often missed
    "meh", "blah", "whatever", "boring", "dull", "bland", "flat", "stale",
    "outdated", "old", "ancient", "obsolete", "behind", "backward",
    "confusing", "unclear", "messy", "cluttered", "chaotic", "disorganized"
}

# Negative phrases that are commonly missed
negative_phrases = {
    "not good", "not great", "not worth", "not recommended", "not helpful",
    "not working", "not satisfied", "not happy", "not impressed", "not clear",
    "could be better", "needs improvement", "not enough", "too much", "too little",
    "waste of time", "waste of money", "not worth it", "save your money",
    "don't buy", "don't recommend", "avoid this", "stay away", "skip this",
    "nothing special", "nothing new", "nothing great", "big disappointment",
    "total waste", "complete waste", "utter waste", "absolute waste",
    "doesn't work", "does not work", "stopped working", "quit working",
    "fell apart", "broke down", "gave up", "worn out", "burned out",
    "rip off", "ripped off", "total scam", "complete scam", "pure scam"
}

# Conditional negative patterns (if-then negatives)
conditional_negatives = {
    "but": ["but terrible", "but awful", "but bad", "but disappointing", "but boring",
           "but useless", "but poor", "but weak", "but slow", "but confusing"],
    "however": ["however terrible", "however bad", "however disappointing"],
    "although": ["although good", "although nice", "although decent"],
    "except": ["except for", "except that"],
    "unfortunately": ["unfortunately"],
    "sadly": ["sadly"],
    "disappointingly": ["disappointingly"]
}

def enhanced_negative_detection(text):
    """Enhanced negative detection with multiple strategies"""
    if not text or not isinstance(text, str):
        return False, None
        
    text_lower = text.lower().strip()
    
    # Strategy 1: Direct negative keyword matching
    words = text_lower.split()
    for word in words:
        if word in negative_keywords:
            return True, f"Negative keyword found: '{word}'"
    
    # Strategy 2: Negative phrase detection
    for phrase in negative_phrases:
        if phrase in text_lower:
            return True, f"Negative phrase found: '{phrase}'"
    
    # Strategy 3: Conditional negative detection (but, however, etc.)
    for condition, neg_patterns in conditional_negatives.items():
        if condition in text_lower:
            for pattern in neg_patterns:
                if pattern in text_lower:
                    return True, f"Conditional negative: '{condition}...{pattern}'"
    
    # Strategy 4: Negation context analysis
    negation_words = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"]
    for i, word in enumerate(words):
        if word in negation_words and i < len(words) - 1:
            # Check next 2-3 words after negation
            next_words = words[i+1:i+4]
            positive_words = ["good", "great", "nice", "excellent", "amazing", "awesome", "love", "like", "best", "perfect"]
            for pos_word in positive_words:
                if pos_word in next_words:
                    return True, f"Negated positive: '{word} {pos_word}'"
    
    # Strategy 5: Low rating indicators
    rating_patterns = [
        r"\b[01]/10\b", r"\b[01]/5\b", r"\b[01] out of 10\b", r"\b[01] out of 5\b",
        r"\b1 star\b", r"\bone star\b", r"\bzero stars\b", r"\b0 stars\b"
    ]
    for pattern in rating_patterns:
        if re.search(pattern, text_lower):
            return True, f"Low rating detected: {pattern}"
    
    # Strategy 6: Length vs content analysis (short dismissive comments)
    if len(words) <= 3:
        dismissive_short = {"meh", "nah", "pass", "skip", "no", "nope", "boring", "lame", "bad"}
        if any(word in dismissive_short for word in words):
            return True, f"Short dismissive comment: '{text}'"
    
    # Strategy 7: Question-based dissatisfaction
    if "?" in text and any(neg in text_lower for neg in ["why", "how", "what"] + list(negative_keywords)):
        if any(neg in text_lower for neg in ["bad", "wrong", "broken", "problem", "issue"]):
            return True, "Negative question pattern detected"
    
    return False, None

def stemming(content):
    """Stem the content with fallback if NLTK unavailable"""
    if not content or not isinstance(content, str):
        return ""
        
    try:
        # Remove non-alphabetic characters
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        
        if NLTK_AVAILABLE:
            # Use NLTK stemming and stopwords
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
        else:
            # Basic fallback - just remove common stopwords without stemming
            stemmed_content = [word for word in stemmed_content if word not in stop_words]
        
        return ' '.join(stemmed_content)
    except Exception as e:
        st.warning(f"Stemming error: {str(e)}")
        return content

def predict_sentiment(comment):
    """Predict sentiment with comprehensive error handling"""
    if not comment or not comment.strip():
        return "Please enter a comment", ""
    
    # Enhanced negative detection
    is_negative, reason = enhanced_negative_detection(comment)
    if is_negative:
        return "Negative", f"Rule-based detection: {reason}"
    
    # Check if models are loaded
    if model is None or vectorizer is None:
        return "Error: Models not loaded", "Please check model files"
    
    # ML model prediction
    try:
        stemmed = stemming(comment)
        if not stemmed.strip():
            return "Neutral", "No meaningful content after preprocessing"
            
        vector = vectorizer.transform([stemmed])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        max_confidence = max(confidence)
        
        # Additional check: if ML says positive but confidence is low, be more cautious
        if sentiment == "Positive" and max_confidence < 0.7:
            # Double-check with more sensitive negative detection
            sensitive_negs = ["okay", "ok", "fine", "decent", "acceptable", "average", "nothing special"]
            if any(neg in comment.lower() for neg in sensitive_negs):
                return "Neutral/Mixed", f"Low confidence positive ({max_confidence:.2f}) with lukewarm language"
        
        return sentiment, f"ML model (Confidence: {max_confidence:.2f})"
    
    except Exception as e:
        return f"Error: {str(e)}", "Model prediction failed"

# Streamlit UI
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS to reduce font size and button spacing
st.markdown("""
    <style>
    /* Reduce text size in buttons */
    h1 {
        font-size: 30px !important;
    }
            
    h3{
        font-size: 20px !important;
    }
            
    .stButton>button {
        font-size: 8px !important;
        padding: 4px 10px !important;
        margin: 2px !important;
    }

    /* Reduce column padding (between buttons) */
    div[data-testid="column"] {
        padding: 0rem 0rem !important;
    }

    /* Reduce vertical space after text area and headings */
    .stTextArea, .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🎬 Enhanced YouTube Comment Sentiment Analysis")


# Check if models are loaded
if model is None or vectorizer is None:
    st.error("⚠️ **Models not loaded!** Please ensure 'trained_model.sav' and 'vectorizer.pkl' are in the same directory.")
    st.info("**Required files:**\n- trained_model.sav (your trained sentiment model)\n- vectorizer.pkl (your text vectorizer)")
    st.stop()

# Initialize session state
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Examples section with problematic cases
st.subheader("🧪 Test Examples (Common False Positives):")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Subtle Negative", key="btn1"):
        st.session_state.text_input = "It's okay I guess, nothing special"

with col2:
    if st.button("Sarcastic", key="btn2"):
        st.session_state.text_input = "Oh wow, this is just perfect... NOT"

with col3:
    if st.button("Conditional Negative", key="btn3"):
        st.session_state.text_input = "Good video but the audio quality is terrible"

with col4:
    if st.button("Clear Negative", key="btn4"):
        st.session_state.text_input = "Complete waste of time, don't recommend"

# Additional examples
col5, col6, col7, col8 = st.columns(4)

with col5:
    if st.button("Negated Positive", key="btn5"):
        st.session_state.text_input = "This is not good at all"

with col6:
    if st.button("Question Negative", key="btn6"):
        st.session_state.text_input = "Why is this so bad? What went wrong?"

with col7:
    if st.button("Positive Example", key="btn7"):
        st.session_state.text_input = "Amazing video! Really loved it, great work!"

with col8:
    if st.button("Clear", key="btn_clear"):
        st.session_state.text_input = ""

st.markdown("---")

# Input section
user_input = st.text_area(
    "💬 Enter a YouTube comment:", 
    value=st.session_state.text_input,
    height=100,
    placeholder="Try entering comments that might be falsely classified as positive...",
    help="Enter any YouTube comment to analyze its sentiment"
)

# Update session state when user types
if user_input != st.session_state.text_input:
    st.session_state.text_input = user_input

# Prediction
col_analyze, col_space = st.columns([1, 3])
with col_analyze:
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if analyze_button and user_input and user_input.strip():
    with st.spinner("Analyzing sentiment..."):
        result, explanation = predict_sentiment(user_input)
        
        # Display result with detailed explanation
        st.markdown("### 📊 Results:")
        
        if result == "Positive":
            st.success(f"**✅ Sentiment: {result}**")
        elif result == "Negative":
            st.error(f"**❌ Sentiment: {result}**")
        elif "Neutral" in result or "Mixed" in result:
            st.warning(f"**⚠️ Sentiment: {result}**")
        else:
            st.info(f"**ℹ️ {result}**")
        
        if explanation:
            st.info(f"**🔍 Method:** {explanation}")

elif analyze_button:
    st.warning("⚠️ Please enter a comment to analyze")

# Debug information
with st.expander("🔧 Debug Information"):
    if user_input:
        st.write("**Current input:**", user_input)
        st.write("**Input length:**", len(user_input), "characters")
        
        is_neg, reason = enhanced_negative_detection(user_input)
        st.write("**Rule-based detection:**", "Negative detected" if is_neg else "No negative patterns found")
        if reason:
            st.write("**Detection reason:**", reason)
        
        # Show stemmed version
        stemmed = stemming(user_input)
        st.write("**Stemmed text:**", stemmed if stemmed else "Empty after preprocessing")
        
        # Show model status
        st.write("**NLTK available:**", "✅ Yes" if NLTK_AVAILABLE else "❌ No")
        st.write("**Models loaded:**", "✅ Yes" if (model is not None and vectorizer is not None) else "❌ No")

# Info section
with st.expander("ℹ️ Enhanced Detection Methods"):
    st.markdown("""
    **🎯 This enhanced version uses multiple strategies to reduce false positives:**
    
    1. **Direct Keyword Matching**: Expanded negative vocabulary
    2. **Phrase Detection**: Multi-word negative expressions  
    3. **Conditional Analysis**: "but", "however", "except" patterns
    4. **Negation Context**: "not good", "don't like" patterns
    5. **Rating Indicators**: "1/10", "zero stars" patterns
    6. **Question Patterns**: Negative questions
    7. **Confidence Thresholding**: Low-confidence positive → neutral
    
    **🎭 Common False Positive Patterns Addressed:**
    - Sarcastic comments
    - Lukewarm/neutral language marked as positive
    - Conditional negatives ("good but...")
    - Negated positives ("not good")
    - Subtle dismissive language
    
    **🔧 Technical Requirements:**
    - `trained_model.sav`: Your trained sentiment classification model
    - `vectorizer.pkl`: Your text vectorization model (TF-IDF, etc.)
    - Optional: NLTK for advanced text preprocessing
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Enhanced sentiment detection with rule-based and ML approaches*")