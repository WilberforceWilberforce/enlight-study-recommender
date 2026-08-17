"""
Data Loading Utilities
Load and process question data from CSV files
"""

import csv
from typing import List, Dict, Tuple


class DataLoader:
    """Loads and processes question data from CSV files."""

    @staticmethod
    def load_questions(filepath: str) -> List[Dict]:
        """
        Load questions from a CSV file.

        Expected columns: id, question_text, topic, difficulty

        Args:
            filepath: Path to CSV file

        Returns:
            List of question dictionaries
        """
        questions = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV file is empty")

                required_fields = {'id', 'question_text', 'topic'}
                if not required_fields.issubset(set(reader.fieldnames)):
                    raise ValueError(f"Missing required fields. Required: {required_fields}")

                for row_num, row in enumerate(reader, start=2):
                    try:
                        question = {
                            'id': row['id'].strip(),
                            'text': row['question_text'].strip(),
                            'topic': row['topic'].strip(),
                            'difficulty': float(row.get('difficulty', 2.5))
                        }

                        if not all([question['id'], question['text'], question['topic']]):
                            raise ValueError("Empty required field")

                        questions.append(question)

                    except (KeyError, ValueError) as e:
                        raise ValueError(f"Error parsing row {row_num}: {e}")

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        return questions

    @staticmethod
    def load_training_data(filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load training data for topic classification.

        Expected columns: question_text, topic

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (texts, labels) for training
        """
        texts = []
        labels = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV file is empty")

                for row in reader:
                    if 'question_text' in row and 'topic' in row:
                        text = row['question_text'].strip()
                        label = row['topic'].strip()

                        if text and label:
                            texts.append(text)
                            labels.append(label)

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        if not texts:
            raise ValueError("No valid training data found in CSV")

        return texts, labels

    @staticmethod
    def validate_data(questions: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate loaded questions data.

        Args:
            questions: List of question dictionaries

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not questions:
            errors.append("No questions loaded")
            return False, errors

        # Check for required fields
        for i, q in enumerate(questions):
            if not q.get('id'):
                errors.append(f"Question {i} missing 'id'")
            if not q.get('text'):
                errors.append(f"Question {i} missing 'text'")
            if not q.get('topic'):
                errors.append(f"Question {i} missing 'topic'")
            if not 1 <= q.get('difficulty', 1) <= 5:
                errors.append(f"Question {i} has invalid difficulty (must be 1-5)")

        # Check for unique IDs
        ids = [q['id'] for q in questions]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate question IDs found")

        # Check for topic variety
        topics = set(q['topic'] for q in questions)
        if len(topics) < 1:
            errors.append("At least one topic required")

        return len(errors) == 0, errors
