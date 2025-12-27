import os
import re
import pickle
import warnings
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Lazy-loaded model and fast-mode fallback
_MODEL = None
_MODEL_LOADED = False
_FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"

POS_WORDS = set([
    "good", "great", "excellent", "amazing", "awesome", "love", "loved", "positive",
    "happy", "benefits", "helpful", "supportive", "flexible", "growth", "friendly",
    "wonderful", "fantastic", "perfect", "satisfied", "enjoy", "enjoyed", "best"
])
NEG_WORDS = set([
    "bad", "terrible", "awful", "hate", "hated", "poor", "negative", "stress",
    "overworked", "overwork", "toxic", "difficult", "slow", "problem", "issue", 
    "meeting", "meetings", "worst", "horrible", "disappointed", "disappointing",
    "frustrated", "frustrating", "unhappy", "complaint", "complaints"
])
NEGATION_WORDS = set([
    "not", "no", "never", "nothing", "nobody", "nowhere", "none", "neither",
    "cannot", "cant", "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt",
    "havent", "hasnt", "hadnt", "isnt", "arent", "wasnt", "werent"
])


def _load_model():
    global _MODEL, _MODEL_LOADED, _FAST_MODE
    if _MODEL_LOADED:
        return _MODEL
    _MODEL_LOADED = True

    if _FAST_MODE:
        return None

    model_path = "sentiment_model.pkl"
    if not os.path.exists(model_path):
        _FAST_MODE = True
        return None

    try:
        # suppress sklearn version warnings during unpickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(model_path, "rb") as f:
                _MODEL = pickle.load(f)
        return _MODEL
    except Exception:
        _FAST_MODE = True
        return None


def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def _rule_based_sentiment(text):
    t = clean_text(text)
    tokens = t.split()
    
    # Check for negations and handle them
    pos_count = 0
    neg_count = 0
    
    # Look for negation patterns
    for i, token in enumerate(tokens):
        # Check if current token is a positive word
        if token in POS_WORDS:
            # Check if there's a negation word nearby (within 2 words before)
            is_negated = False
            for j in range(max(0, i-2), i):
                if tokens[j] in NEGATION_WORDS:
                    is_negated = True
                    break
            
            if is_negated:
                # Negated positive word counts as negative
                neg_count += 1
            else:
                pos_count += 1
        
        # Check if current token is a negative word
        elif token in NEG_WORDS:
            neg_count += 1
        
        # Check if current token is a negation word followed by positive context
        elif token in NEGATION_WORDS:
            # If negation word appears, it's generally negative
            neg_count += 0.5
    
    # Also check for phrases like "no facilities", "not good", etc.
    text_lower = t.lower()
    if any(phrase in text_lower for phrase in ["no such", "no facilities", "not feeling", "not good", "not great", "not happy"]):
        neg_count += 1
    
    if pos_count == 0 and neg_count == 0:
        return 0.0, 'Neutral'
    
    polarity = (pos_count - neg_count) / max(1, pos_count + neg_count)
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return round(float(polarity), 3), label


def get_polarity_and_sentiment(text):
    global _FAST_MODE
    model = _load_model()
    if model is None:
        return _rule_based_sentiment(text)

    try:
        pred = model.predict([text])[0]
    except Exception:
        # fallback to rule-based if model prediction fails
        return _rule_based_sentiment(text)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_)
            p_pos = proba[classes.index("positive")] if "positive" in classes else 0.0
            p_neg = proba[classes.index("negative")] if "negative" in classes else 0.0
            polarity = p_pos - p_neg
        except Exception:
            polarity = 0.0
    else:
        if str(pred).lower() == "positive":
            polarity = 1.0
        elif str(pred).lower() == "negative":
            polarity = -1.0
        else:
            polarity = 0.0

    label = str(pred).capitalize()
    return round(float(polarity), 3), label


def analyze_feedback_list(feedback_list):
    results = []
    for fb in feedback_list:
        if not fb.strip():
            continue
        cleaned = clean_text(fb)
        polarity, sentiment = get_polarity_and_sentiment(fb)
        results.append({
            "original": fb.strip(),
            "cleaned": cleaned,
            "polarity": polarity,
            "sentiment": sentiment
        })

    if not results:
        return pd.DataFrame(), []

    df = pd.DataFrame(results)

    # Extract top keywords using CountVectorizer with counts
    vectorizer = CountVectorizer(max_features=10, stop_words='english')
    try:
        X = vectorizer.fit_transform(df["cleaned"])  # shape (n_samples, n_features)
        counts = X.toarray().sum(axis=0)
        words = vectorizer.get_feature_names_out()
        pairs = list(zip(words.tolist(), counts.tolist()))
        # sort by count desc
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        keywords = pairs
    except Exception:
        keywords = []

    return df, keywords
