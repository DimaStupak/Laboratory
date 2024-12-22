import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')

with open('norton.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

def process_tweet(tweet):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    tweet = re.sub(r'http\S+|www\.\S+', '', tweet)  
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  
    tweet = re.sub(r'\s+', ' ', tweet).strip().lower()  

    tokens = tweet.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return tokens

processed_tweets = [process_tweet(tweet) for tweet in tweets]

labels = [0] * (len(processed_tweets) // 2) + [1] * (len(processed_tweets) // 2)
labels = labels[:len(processed_tweets)]

data = pd.DataFrame({'text': [' '.join(tweet) for tweet in processed_tweets], 'label': labels})
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

def build_frequency_dictionary(tweets, labels):
    frequency_dict = {}
    for label, tweet in zip(labels, tweets):
        for word in tweet:
            pair = (word, label)
            frequency_dict[pair] = frequency_dict.get(pair, 0) + 1
    return frequency_dict

train_tokens = [process_tweet(tweet) for tweet in X_train]
frequency_dict = build_frequency_dictionary(train_tokens, y_train)

log_prior_positive = np.log(sum(y_train) / len(y_train))
log_prior_negative = np.log(1 - sum(y_train) / len(y_train))

def compute_log_likelihood(frequency_dict, train_tokens, y_train):
    word_counts = Counter(word for tokens in train_tokens for word in tokens)
    vocab_size = len(word_counts)

    positive_word_counts = sum(count for (word, label), count in frequency_dict.items() if label == 1)
    negative_word_counts = sum(count for (word, label), count in frequency_dict.items() if label == 0)

    log_likelihood = {}
    for word in word_counts:
        count_positive = frequency_dict.get((word, 1), 0)
        count_negative = frequency_dict.get((word, 0), 0)

        log_likelihood[word] = {
            1: np.log((count_positive + 1) / (positive_word_counts + vocab_size)),
            0: np.log((count_negative + 1) / (negative_word_counts + vocab_size))
        }
    return log_likelihood

log_likelihood = compute_log_likelihood(frequency_dict, train_tokens, y_train)

def naive_bayes_predict(tweet, log_prior_positive, log_prior_negative, log_likelihood):
    tokens = process_tweet(tweet)
    positive_score = log_prior_positive
    negative_score = log_prior_negative

    for word in tokens:
        if word in log_likelihood:
            positive_score += log_likelihood[word][1]
            negative_score += log_likelihood[word][0]

    return 1 if positive_score > negative_score else 0

y_pred = [naive_bayes_predict(tweet, log_prior_positive, log_prior_negative, log_likelihood) for tweet in X_test]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def analyze_words(log_likelihood):
    word_ratios = {word: log_likelihood[word][1] - log_likelihood[word][0] for word in log_likelihood}
    most_positive = sorted(word_ratios.items(), key=lambda x: x[1], reverse=True)[:10]
    most_negative = sorted(word_ratios.items(), key=lambda x: x[1])[:10]
    return most_positive, most_negative

most_positive, most_negative = analyze_words(log_likelihood)
print("positive words:", most_positive)
print("negative words:", most_negative)

custom_tweet = "I absolutely love this amazing product!"
prediction = naive_bayes_predict(custom_tweet, log_prior_positive, log_prior_negative, log_likelihood)
print(f"Custom tweet sentiment: {'Positive' if prediction == 1 else 'Negative'}")