import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------
# LOAD THE DATASET
# --------------------------------

fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")


# --------------------------------
# ADD LABELS
# --------------------------------

fake_news["label"] = 0
real_news["label"] = 1


# --------------------------------
# COMBINE THE DATA
# --------------------------------

data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)


# --------------------------------
# USE NEWS TEXT
# --------------------------------

# Most Fake/True news datasets contain
# a column called "title".
X = data["title"].fillna("")
y = data["label"]


# --------------------------------
# CONVERT TEXT INTO NUMBERS
# --------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_vectorized = vectorizer.fit_transform(X)


# --------------------------------
# TRAIN THE AI MODEL
# --------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_vectorized, y)


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def predict_news(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = max(probabilities) * 100


    if prediction == 1:
        result = "REAL"
    else:
        result = "POSSIBLY FAKE"
      

    return result, round(confidence, 2)
