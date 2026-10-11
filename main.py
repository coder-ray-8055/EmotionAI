import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'train.txt')

df = pd.read_csv(data_path, sep = ';', header = None, names=['text', 'emotion'])
# print(df.head())

# Record original emotion counts and total samples for metadata
emotion_counts = df['emotion'].value_counts().to_dict()
total_samples = len(df)

unique_emotons = (df['emotion'].unique())
emotion_numbers = {}
i= 0
for emo in unique_emotons:
    emotion_numbers[emo] = i
    i+=1
df['emotion'] = df['emotion'].map(emotion_numbers)

# Create reverse mapping: number -> emotion name
emotion_map = {num: emo for emo, num in emotion_numbers.items()}

df['text'] = df['text'].apply(lambda x : x.lower())

def remove_punc(txt):
    return txt.translate(str.maketrans('','',string.punctuation ))

df['text'] = df['text'].apply(remove_punc)

def remove_numbers(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new
df['text'] = df['text'].apply(remove_numbers)

def remove_emojis(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new = new + i
    return new
df['text'] = df['text'].apply(remove_emojis)

# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(len(stop_words))

def remove(txt):
    words = word_tokenize(txt)
    cleaned = []
    for i in words:
        if not i in stop_words:
            cleaned.append(i)
    return ' '.join(cleaned)
df['text'] = df['text'].apply(remove)
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.20, random_state=42)

bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

nb_model= MultinomialNB()
nb_model.fit(X_train_bow, y_train)
pred_bow = nb_model.predict(X_test_bow)
bow_accuracy = float(accuracy_score(y_test, pred_bow))
print(bow_accuracy)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
nb2_model = MultinomialNB()
nb2_model.fit(X_train_tfidf, y_train)
y_pred = nb2_model.predict(X_test_tfidf)
tfidf_accuracy = float(accuracy_score(y_test, y_pred))
print(tfidf_accuracy)

# Save model, vectorizer, and metadata using pickle
metadata = {
    'emotion_map': emotion_map,
    'bow_accuracy': bow_accuracy,
    'tfidf_accuracy': tfidf_accuracy,
    'total_samples': total_samples,
    'emotion_counts': emotion_counts
}

with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
    # Save the better model — BoW outperforms TF-IDF on this dataset
    pickle.dump(nb_model, f)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(bow_vectorizer, f)

with open(os.path.join(BASE_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

print("Saved model.pkl, vectorizer.pkl, and metadata.pkl successfully.")