"""
Tests for Study Recommender
"""

import pytest
from src.recommender import StudyRecommender
from src.classifier import QuestionClassifier


class TestStudyRecommender:
    """Test cases for StudyRecommender."""

    @pytest.fixture
    def trained_recommender(self):
        """Create a trained recommender."""
        recommender = StudyRecommender()

        # Train classifier
        texts = [
            "What is calculus?", "Calculate the derivative",
            "What is geometry?", "Find the area of a circle",
        ]
        labels = ["calculus", "calculus", "geometry", "geometry"]

        recommender.classifier.train(texts, labels)

        # Add questions
        questions = [
            {
                'id': '1',
                'text': 'What is the derivative of x?',
                'topic': 'calculus',
                'difficulty': 2.0
            },
            {
                'id': '2',
                'text': 'Calculate the integral',
                'topic': 'calculus',
                'difficulty': 3.5
            },
            {
                'id': '3',
                'text': 'Find the area of a triangle',
                'topic': 'geometry',
                'difficulty': 1.5
            },
            {
                'id': '4',
                'text': 'Calculate volume of sphere',
                'topic': 'geometry',
                'difficulty': 3.0
            },
        ]
        recommender.add_questions(questions)

        return recommender

    def test_recommender_initialization(self):
        """Test recommender initializes correctly."""
        recommender = StudyRecommender()
        assert recommender.classifier is not None
        assert len(recommender.questions_db) == 0
        assert len(recommender.student_accuracy) == 0

    def test_add_questions(self, trained_recommender):
        """Test adding questions."""
        assert len(trained_recommender.questions_db) == 4

    def test_add_questions_validation(self):
        """Test question validation."""
        recommender = StudyRecommender()

        invalid_questions = [
            {'id': '1', 'text': 'test'},  # Missing topic and difficulty
        ]

        with pytest.raises(ValueError):
            recommender.add_questions(invalid_questions)

    def test_get_questions_by_topic(self, trained_recommender):
        """Test getting questions by topic."""
        calculus_q = trained_recommender.get_questions_by_topic('calculus', count=2)

        assert len(calculus_q) == 2
        assert all(q['topic'] == 'calculus' for q in calculus_q)

    def test_get_questions_by_topic_with_difficulty(self, trained_recommender):
        """Test filtering by difficulty."""
        easy_questions = trained_recommender.get_questions_by_topic(
            'geometry', difficulty=1.5, count=5
        )

        assert len(easy_questions) > 0
        assert all(1 <= q['difficulty'] <= 3 for q in easy_questions)

    def test_classify_question(self, trained_recommender):
        """Test question classification."""
        result = trained_recommender.classify_question("What is the derivative?")
        assert result in ['calculus', 'geometry']

    def test_classify_with_confidence(self, trained_recommender):
        """Test classification with confidence."""
        probs = trained_recommender.classify_with_confidence("What is calculus?")

        assert isinstance(probs, dict)
        assert 'calculus' in probs or 'geometry' in probs

    def test_update_student_accuracy(self, trained_recommender):
        """Test accuracy updates."""
        trained_recommender.update_student_accuracy('calculus', 0.85)

        assert trained_recommender.student_accuracy['calculus'] == 0.85

    def test_accuracy_validation(self, trained_recommender):
        """Test accuracy value validation."""
        with pytest.raises(ValueError):
            trained_recommender.update_student_accuracy('calculus', 1.5)

        with pytest.raises(ValueError):
            trained_recommender.update_student_accuracy('calculus', -0.1)

    def test_adapt_difficulty(self, trained_recommender):
        """Test difficulty adaptation."""
        # Excellent performance
        trained_recommender.update_student_accuracy('calculus', 0.95)
        diff = trained_recommender.adapt_difficulty('calculus')
        assert diff > 4.0

        # Poor performance
        trained_recommender.update_student_accuracy('geometry', 0.3)
        diff = trained_recommender.adapt_difficulty('geometry')
        assert diff < 2.0

    def test_get_next_question(self, trained_recommender):
        """Test getting next question."""
        question = trained_recommender.get_next_question('calculus')
        assert question is not None
        assert 'id' in question
        assert 'text' in question
        assert 'topic' in question

    def test_get_study_plan(self, trained_recommender):
        """Test study plan generation."""
        plan = trained_recommender.get_study_plan(['calculus', 'geometry'], num_questions=3)

        assert len(plan) <= 3
        assert all('id' in q and 'text' in q for q in plan)

    def test_performance_summary(self, trained_recommender):
        """Test performance summary."""
        trained_recommender.update_student_accuracy('calculus', 0.8)

        summary = trained_recommender.get_performance_summary()

        assert 'calculus' in summary
        assert 'geometry' in summary
        assert summary['calculus']['accuracy'] == 0.8
        assert summary['geometry']['accuracy'] is None
