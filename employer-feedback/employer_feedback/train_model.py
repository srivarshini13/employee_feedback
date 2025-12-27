# train_model.py  (for glassdoor-companies-reviews.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

CSV_PATH = "glassdoor-companies-reviews.csv"

print(f"Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print("Columns found:", list(df.columns))

# --------- 1. AUTO-DETECT TEXT COLUMN ---------
text_candidates = [
    "review_text", "review", "text", "summary",
    "pros", "cons", "feedback", "comment", "body", "description"
]

TEXT_COL = None
for col in text_candidates:
    if col in df.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    raise SystemExit(
        "Could not find a text column. Please rename your review column "
        "to something like 'review_text' or update text_candidates in train_model.py."
    )

print(f"Using text column: {TEXT_COL}")

# --------- 2. FIND SENTIMENT OR RATING COLUMN ---------
label_col = None
rating_col = None

# Case 1: explicit sentiment/label column
for cand in ["sentiment", "label", "polarity"]:
    if cand in df.columns:
        label_col = cand
        break

# Case 2: no sentiment column → look for rating
if label_col is None:
    rating_candidates = [
        "rating", "overall_rating", "overall", "rating_overall",
        "stars", "score", "overall_ratings"
    ]
    for col in rating_candidates:
        if col in df.columns:
            rating_col = col
            break

    if rating_col is None:
        raise SystemExit(
            "Could not find sentiment or rating column. "
            "Please create one named 'sentiment' or a numeric 'rating/overall_rating'."
        )

    print(f"Using numeric rating column: {rating_col} → will convert to sentiment labels.")

    def rating_to_sentiment(r):
        try:
            val = float(r)
        except Exception:
            return "neutral"
        # You can tweak these thresholds if needed
        if val <= 2:
            return "negative"
        elif val == 3:
            return "neutral"
        else:  # 4 or 5
            return "positive"

    df["sentiment"] = df[rating_col].apply(rating_to_sentiment)
    LABEL_COL = "sentiment"
else:
    LABEL_COL = label_col
    print(f"Using existing sentiment column: {LABEL_COL}")

# --------- 3. CLEAN AND PREPARE DATA ---------
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)
df = df[df[TEXT_COL].str.strip() != ""]

X = df[TEXT_COL]
y = df[LABEL_COL].astype(str)

print(f"Total usable samples: {len(df)}")
print("\nLabel distribution:")
print(y.value_counts())

# --------- 4. TRAIN / TEST SPLIT (with safe fallback) ---------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # if class counts are too small for stratify, just don't stratify
    print("\nWarning: could not stratify split (probably small or skewed data).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

# --------- 5. BUILD MODEL PIPELINE ---------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    # class_weight='balanced' helps when neutral vs negative is imbalanced
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# --------- 6. TRAIN ---------
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# --------- 7. EVALUATE ---------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.3f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))

# --------- 8. SAVE MODEL ---------
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved as sentiment_model.pkl")
