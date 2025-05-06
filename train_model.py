import os

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("news.csv")  # Dataset with 'text' and 'label' columns

# Split data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("model/vectorizer.pkl", "wb"))
