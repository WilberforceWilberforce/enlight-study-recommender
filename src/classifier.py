"""
Naïve Bayes Text Classifier for Study Questions
Classifies questions by topic using bag-of-words features
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from typing import List, Tuple, Dict


class QuestionClassifier:
    """Classifies study questions by topic using Naïve Bayes."""

    def __init__(self):
        self.classifier = MultinomialNB()
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            min_df=1,
            max_df=0.9
        )
        self.trained = False
        self.classes_ = None

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Train the classifier on labeled text data.

        Args:
            texts: List of question texts
            labels: List of topic labels

        Returns:
            Dictionary with training statistics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts must match number of labels")

        if len(set(labels)) < 2:
            raise ValueError("Must have at least 2 different topics for training")

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.trained = True
        self.classes_ = self.classifier.classes_

        # Calculate training accuracy
        predictions = self.classifier.predict(X)
        accuracy = np.mean(predictions == np.array(labels))

        return {
            'accuracy': accuracy,
            'num_samples': len(texts),
            'num_features': len(self.vectorizer.get_feature_names_out()),
            'num_topics': len(set(labels))
        }

    def predict(self, text: str) -> str:
        """
        Predict the topic/label for a given text.

        Args:
            text: Question text to classify

        Returns:
            Predicted topic label
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before prediction")

        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get probability distribution for all topics.

        Args:
            text: Question text to classify

        Returns:
            Dictionary with topic probabilities
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before prediction")

        X = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(X)[0]

        return {
            topic: float(prob)
            for topic, prob in zip(self.classes_, probabilities)
        }

    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict topics for multiple texts.

        Args:
            texts: List of question texts

        Returns:
            List of predicted topic labels
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before prediction")

        X = self.vectorizer.transform(texts)
        return list(self.classifier.predict(X))

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'vectorizer': self.vectorizer,
                'classes': self.classes_
            }, f)

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.

        Args:
            filepath: Path to load the model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.vectorizer = data['vectorizer']
            self.classes_ = data['classes']
            self.trained = True

    def get_feature_importance(self, topic: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for a given topic.

        Args:
            topic: Topic label to analyze
            top_n: Number of top features to return

        Returns:
            List of (feature, importance) tuples
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before analysis")

        if topic not in self.classes_:
            raise ValueError(f"Topic '{topic}' not found in training data")

        feature_names = self.vectorizer.get_feature_names_out()
        topic_idx = list(self.classes_).index(topic)

        # Get log probabilities for this topic
        log_probs = self.classifier.feature_log_prob_[topic_idx]

        # Get top features
        top_indices = np.argsort(log_probs)[-top_n:][::-1]
        top_features = [
            (feature_names[i], float(log_probs[i]))
            for i in top_indices
        ]

        return top_features
