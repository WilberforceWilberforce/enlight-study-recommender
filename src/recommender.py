"""
Study Question Recommender
Recommends questions based on topic and adapts difficulty based on performance
"""

from .classifier import QuestionClassifier
from typing import List, Dict, Tuple, Optional
import random


class StudyRecommender:
    """Recommends study questions based on topic and difficulty."""

    def __init__(self, model_path: Optional[str] = None):
        self.classifier = QuestionClassifier()
        if model_path:
            self.classifier.load_model(model_path)

        # Store question database
        self.questions_db = {}  # {id: {text, topic, difficulty, difficulty_range}}
        self.student_accuracy = {}  # {topic: accuracy}

    def add_questions(self, questions: List[Dict]) -> None:
        """
        Add questions to the recommendation database.

        Args:
            questions: List of question dicts with 'id', 'text', 'topic', 'difficulty'
        """
        for q in questions:
            if not all(k in q for k in ['id', 'text', 'topic', 'difficulty']):
                raise ValueError("Each question must have id, text, topic, and difficulty")

            self.questions_db[q['id']] = {
                'text': q['text'],
                'topic': q['topic'],
                'difficulty': q['difficulty'],  # 1-5 scale
                'difficulty_range': q.get('difficulty_range', [q['difficulty'] - 0.5, q['difficulty'] + 0.5])
            }

    def get_questions_by_topic(self, topic: str,
                               difficulty: Optional[float] = None,
                               count: int = 5) -> List[Dict]:
        """
        Get questions for a specific topic.

        Args:
            topic: Topic to get questions for
            difficulty: Target difficulty (1-5). If None, returns mixed
            count: Number of questions to return

        Returns:
            List of recommended questions
        """
        # Filter by topic
        matching = [
            (qid, q) for qid, q in self.questions_db.items()
            if q['topic'].lower() == topic.lower()
        ]

        if not matching:
            return []

        # Filter by difficulty if specified
        if difficulty is not None:
            difficulty = max(1.0, min(5.0, difficulty))  # Clamp to 1-5
            matching = [
                (qid, q) for qid, q in matching
                if abs(q['difficulty'] - difficulty) <= 1.0
            ]

        # Return random sample
        selected = random.sample(matching, min(count, len(matching)))
        return [
            {
                'id': qid,
                'text': q['text'],
                'topic': q['topic'],
                'difficulty': q['difficulty']
            }
            for qid, q in selected
        ]

    def classify_question(self, question_text: str) -> str:
        """
        Classify a question to a topic.

        Args:
            question_text: The question to classify

        Returns:
            Predicted topic
        """
        if not self.classifier.trained:
            raise ValueError("Classifier must be trained before use")

        return self.classifier.predict(question_text)

    def classify_with_confidence(self, question_text: str) -> Dict[str, float]:
        """
        Get topic probabilities for a question.

        Args:
            question_text: The question to classify

        Returns:
            Dictionary with topic confidences
        """
        if not self.classifier.trained:
            raise ValueError("Classifier must be trained before use")

        return self.classifier.predict_proba(question_text)

    def update_student_accuracy(self, topic: str, accuracy: float) -> None:
        """
        Update student accuracy for a topic.

        Args:
            topic: Topic to update
            accuracy: Accuracy score (0-1)
        """
        if not 0 <= accuracy <= 1:
            raise ValueError("Accuracy must be between 0 and 1")

        self.student_accuracy[topic] = accuracy

    def adapt_difficulty(self, topic: str) -> float:
        """
        Adapt question difficulty based on student performance.

        Args:
            topic: Topic to adapt difficulty for

        Returns:
            Recommended difficulty (1-5)
        """
        accuracy = self.student_accuracy.get(topic, 0.5)

        # Difficulty scaling based on performance
        if accuracy >= 0.9:
            # Excellent performance - increase difficulty
            return 4.5
        elif accuracy >= 0.8:
            # Good performance - increase difficulty moderately
            return 3.5
        elif accuracy >= 0.7:
            # Acceptable performance - maintain difficulty
            return 2.5
        elif accuracy >= 0.5:
            # Struggling - reduce difficulty
            return 2.0
        else:
            # Very low performance - significantly reduce difficulty
            return 1.5

    def get_next_question(self, topic: str, auto_difficulty: bool = True) -> Optional[Dict]:
        """
        Get the next recommended question for a student.

        Args:
            topic: Topic to get question for
            auto_difficulty: If True, adapt difficulty based on performance

        Returns:
            Recommended question or None if no questions available
        """
        difficulty = self.adapt_difficulty(topic) if auto_difficulty else 3.0

        questions = self.get_questions_by_topic(topic, difficulty, count=1)

        if questions:
            return questions[0]
        return None

    def get_study_plan(self, topics: List[str],
                      num_questions: int = 10) -> List[Dict]:
        """
        Generate a study plan across multiple topics.

        Args:
            topics: List of topics to study
            num_questions: Total number of questions to include

        Returns:
            Ordered list of recommended questions
        """
        plan = []
        questions_per_topic = num_questions // len(topics) if topics else 0

        for topic in topics:
            difficulty = self.adapt_difficulty(topic)
            questions = self.get_questions_by_topic(topic, difficulty, questions_per_topic)
            plan.extend(questions)

        # Shuffle for variety while maintaining some topic grouping
        random.shuffle(plan)
        return plan[:num_questions]

    def get_performance_summary(self) -> Dict[str, Dict]:
        """
        Get student performance summary across topics.

        Returns:
            Dictionary with performance metrics by topic
        """
        summary = {}

        for topic in set(q['topic'] for q in self.questions_db.values()):
            accuracy = self.student_accuracy.get(topic, None)
            difficulty = self.adapt_difficulty(topic) if accuracy else None

            summary[topic] = {
                'accuracy': accuracy,
                'recommended_difficulty': difficulty,
                'num_questions_available': sum(
                    1 for q in self.questions_db.values()
                    if q['topic'] == topic
                )
            }

        return summary
