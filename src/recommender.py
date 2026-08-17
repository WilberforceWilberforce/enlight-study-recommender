# Study Question Recommender

from .classifier import QuestionClassifier


class StudyRecommender:
    """Recommends study questions based on topic and difficulty."""

    def __init__(self, model_path=None):
        self.classifier = QuestionClassifier()
        if model_path:
            self.classifier.load_model(model_path)

    def get_questions(self, topic, difficulty=None):
        """Get recommended questions for a given topic and difficulty level."""
        # Placeholder implementation
        return []

    def adapt_difficulty(self, student_accuracy):
        """Adapt question difficulty based on student performance."""
        if student_accuracy >= 0.8:
            return "hard"
        elif student_accuracy >= 0.6:
            return "medium"
        else:
            return "easy"
