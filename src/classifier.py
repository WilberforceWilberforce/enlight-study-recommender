# Naïve Bayes Text Classifier for Study Questions

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class QuestionClassifier:
    """Classifies study questions by topic using Naïve Bayes."""

    def __init__(self):
        self.classifier = MultinomialNB()
        self.vectorizer = CountVectorizer()

    def train(self, texts, labels):
        """Train the classifier on labeled text data."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def predict(self, text):
        """Predict the topic/label for a given text."""
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]

    def save_model(self, filepath):
        """Save trained model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'classifier': self.classifier, 'vectorizer': self.vectorizer}, f)

    def load_model(self, filepath):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.vectorizer = data['vectorizer']
