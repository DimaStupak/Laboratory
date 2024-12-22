import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from collections import Counter

with open('norton.txt', 'r', encoding='utf-8') as file:
    texts = file.readlines()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

texts_cleaned = [preprocess_text(text) for text in texts]

print(f"Кількість строк: {len(texts_cleaned)}")

labels = [0] * (len(texts_cleaned) // 2) + [1] * (len(texts_cleaned) // 2)
labels = labels[:len(texts_cleaned)]

data = pd.DataFrame({'text': texts_cleaned, 'label': labels})

vectorizer = CountVectorizer()
X = data['text']
y = data['label']
X_vectorized = vectorizer.fit_transform(X)

positive_texts = data[data['label'] == 1]['text']
negative_texts = data[data['label'] == 0]['text']

positive_words = Counter(" ".join(positive_texts).split())
negative_words = Counter(" ".join(negative_texts).split())

print("Словник частотності для позитивного класу (топ 10):")
print(positive_words.most_common(10))

print("\nСловник частотності для негативного класу (топ 10):")
print(negative_words.most_common(10))

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
