"""
Tests for Question Classifier
"""

import pytest
from src.classifier import QuestionClassifier


class TestQuestionClassifier:
    """Test cases for QuestionClassifier."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        texts = [
            "What is the derivative of x squared?",
            "Calculate the integral of cosine x",
            "Solve this differential equation",
            "What is the Pythagorean theorem?",
            "How do you find the area of a triangle?",
            "Calculate the volume of a sphere",
        ]
        labels = [
            "calculus", "calculus", "calculus",
            "geometry", "geometry", "geometry"
        ]
        return texts, labels

    @pytest.fixture
    def trained_classifier(self, sample_data):
        """Create a trained classifier."""
        classifier = QuestionClassifier()
        texts, labels = sample_data
        classifier.train(texts, labels)
        return classifier

    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        classifier = QuestionClassifier()
        assert not classifier.trained
        assert classifier.classifier is not None
        assert classifier.vectorizer is not None

    def test_training(self, sample_data):
        """Test classifier training."""
        classifier = QuestionClassifier()
        texts, labels = sample_data

        stats = classifier.train(texts, labels)

        assert classifier.trained
        assert 'accuracy' in stats
        assert 'num_samples' in stats
        assert stats['num_samples'] == 6
        assert stats['num_topics'] == 2

    def test_training_validation(self, sample_data):
        """Test training input validation."""
        classifier = QuestionClassifier()
        texts, labels = sample_data

        # Test mismatched lengths
        with pytest.raises(ValueError):
            classifier.train(texts[:-1], labels)

        # Test insufficient topics
        with pytest.raises(ValueError):
            classifier.train(texts, ["same"] * len(texts))

    def test_prediction(self, trained_classifier):
        """Test single prediction."""
        result = trained_classifier.predict("What is the derivative?")
        assert result in ["calculus", "geometry"]

    def test_prediction_requires_training(self):
        """Test that untrained classifier cannot predict."""
        classifier = QuestionClassifier()
        with pytest.raises(ValueError):
            classifier.predict("test question")

    def test_predict_proba(self, trained_classifier):
        """Test probability predictions."""
        probs = trained_classifier.predict_proba("What is calculus?")

        assert isinstance(probs, dict)
        assert "calculus" in probs
        assert "geometry" in probs
        assert 0 <= probs["calculus"] <= 1
        assert 0 <= probs["geometry"] <= 1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_batch_prediction(self, trained_classifier):
        """Test batch prediction."""
        texts = [
            "What is the derivative?",
            "Calculate the area",
        ]
        results = trained_classifier.predict_batch(texts)

        assert len(results) == 2
        assert all(r in ["calculus", "geometry"] for r in results)

    def test_feature_importance(self, trained_classifier):
        """Test feature importance analysis."""
        importance = trained_classifier.get_feature_importance("calculus", top_n=5)

        assert isinstance(importance, list)
        assert len(importance) <= 5
        assert all(isinstance(f, tuple) and len(f) == 2 for f in importance)

    def test_save_and_load_model(self, trained_classifier, tmp_path):
        """Test model persistence."""
        model_path = tmp_path / "model.pkl"

        # Save
        trained_classifier.save_model(str(model_path))
        assert model_path.exists()

        # Load into new classifier
        new_classifier = QuestionClassifier()
        new_classifier.load_model(str(model_path))

        # Verify it works
        result = new_classifier.predict("What is the derivative?")
        assert result in ["calculus", "geometry"]

    def test_save_untrained_model(self):
        """Test that untrained model cannot be saved."""
        classifier = QuestionClassifier()
        with pytest.raises(ValueError):
            classifier.save_model("dummy.pkl")
