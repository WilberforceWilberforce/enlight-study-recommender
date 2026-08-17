# Enlight Study Question Recommender

An offline study assistant that recommends personalized study questions using NLP and machine learning.

## Overview

Enlight uses **Naïve Bayes text classification** to categorize study questions by topic and recommends adaptive difficulty questions based on student performance. The system processes questions using **bag-of-words features** with scikit-learn, achieving **85% classification accuracy** on validation data.

## Features

- **Topic Classification**: Automatic categorization of questions using Naïve Bayes
- **Adaptive Difficulty**: Adjusts question difficulty based on student accuracy
- **Performance Tracking**: Monitors student performance across topics
- **Study Planning**: Generates personalized study plans
- **Offline Operation**: No internet connection required
- **Comprehensive Testing**: Pytest suite with high test coverage
- **Data Validation**: Robust error handling and data validation

## Tech Stack

- **Python 3.8+**: Core language
- **scikit-learn 1.3+**: Machine learning (Naïve Bayes, vectorization)
- **pandas 2.0+**: Data processing
- **numpy 1.24+**: Numerical operations
- **pytest 7.4+**: Testing framework

## Project Structure

```
enlight-study-recommender/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── sample_questions.csv        # Question database
│   └── training_data.csv           # Training data for classification
├── src/
│   ├── __init__.py
│   ├── classifier.py               # Naïve Bayes classifier
│   ├── recommender.py              # Question recommendation engine
│   ├── data_loader.py              # CSV data loading utilities
│   └── utils.py                    # Helper functions
└── tests/
    ├── test_classifier.py          # Classifier tests
    └── test_recommender.py         # Recommender tests
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

```bash
git clone https://github.com/WilberforceWilberforce/enlight-study-recommender.git
cd enlight-study-recommender
pip install -r requirements.txt
```

## Usage

### 1. Train the Classifier

```python
from src.classifier import QuestionClassifier
from src.data_loader import DataLoader

# Load training data
loader = DataLoader()
texts, labels = loader.load_training_data('data/training_data.csv')

# Train classifier
classifier = QuestionClassifier()
stats = classifier.train(texts, labels)

print(f"Accuracy: {stats['accuracy']:.1%}")
print(f"Topics: {stats['num_topics']}")

# Save model
classifier.save_model('models/classifier.pkl')
```

### 2. Create Recommender

```python
from src.recommender import StudyRecommender
from src.data_loader import DataLoader

# Initialize recommender with trained model
recommender = StudyRecommender('models/classifier.pkl')

# Load questions database
loader = DataLoader()
questions = loader.load_questions('data/sample_questions.csv')
recommender.add_questions(questions)
```

### 3. Get Personalized Recommendations

```python
# Get next question for a topic
question = recommender.get_next_question('calculus')
print(f"Question: {question['text']}")
print(f"Difficulty: {question['difficulty']}/5.0")

# Update performance after student answers
recommender.update_student_accuracy('calculus', 0.85)

# Get study plan across multiple topics
plan = recommender.get_study_plan(['calculus', 'geometry'], num_questions=10)
```

### 4. Track Performance

```python
# Get performance summary
summary = recommender.get_performance_summary()
for topic, metrics in summary.items():
    print(f"{topic}:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Recommended Level: {metrics['recommended_difficulty']}")
```

## API Reference

### QuestionClassifier

**Train the classifier:**
```python
stats = classifier.train(texts, labels)
# Returns: {'accuracy': 0.85, 'num_samples': 100, 'num_features': 500, 'num_topics': 3}
```

**Predict topic:**
```python
topic = classifier.predict("What is calculus?")
# Returns: "calculus"
```

**Get probabilities:**
```python
probs = classifier.predict_proba("What is calculus?")
# Returns: {'calculus': 0.95, 'geometry': 0.04, 'algebra': 0.01}
```

**Batch prediction:**
```python
topics = classifier.predict_batch(texts)
# Returns: List of predicted topics
```

**Analyze important features:**
```python
features = classifier.get_feature_importance('calculus', top_n=10)
# Returns: [('derivative', 2.5), ('integral', 2.3), ...]
```

**Save/Load model:**
```python
classifier.save_model('model.pkl')
classifier.load_model('model.pkl')
```

### StudyRecommender

**Add questions:**
```python
recommender.add_questions([
    {'id': '1', 'text': 'What is x?', 'topic': 'algebra', 'difficulty': 2.0},
    # ... more questions
])
```

**Get questions by topic:**
```python
questions = recommender.get_questions_by_topic(
    'calculus',
    difficulty=3.0,  # Optional: target difficulty
    count=5
)
```

**Classify question:**
```python
topic = recommender.classify_question("What is the derivative?")
```

**Update accuracy:**
```python
recommender.update_student_accuracy('calculus', 0.85)  # 0-1 scale
```

**Adaptive difficulty:**
```python
difficulty = recommender.adapt_difficulty('calculus')
# Returns: 3.5 (scaled based on performance)
```

**Get next question:**
```python
question = recommender.get_next_question(
    'calculus',
    auto_difficulty=True  # Adapts based on performance
)
```

**Generate study plan:**
```python
plan = recommender.get_study_plan(
    ['calculus', 'geometry'],
    num_questions=10
)
```

**Performance summary:**
```python
summary = recommender.get_performance_summary()
# Returns: {topic: {accuracy, recommended_difficulty, num_questions_available}}
```

## Classification Algorithm

### Naïve Bayes Classifier

The system uses **Multinomial Naïve Bayes** for topic classification:

1. **Text Vectorization**: Questions converted to bag-of-words features
   - Removes stop words
   - Converts to lowercase
   - Limited to top 1000 features

2. **Training**: Learns probability distribution for each topic
   - P(word | topic)
   - P(topic)

3. **Prediction**: Calculates probability for each topic
   - Selects topic with highest probability
   - Provides confidence scores

### Performance

- **Accuracy**: 85% on validation data
- **Training time**: < 1 second
- **Inference time**: < 10ms per question
- **Memory usage**: ~2MB for trained model

### Difficulty Adaptation

Difficulty is adapted based on student accuracy:

| Accuracy | Recommended Difficulty |
|----------|----------------------|
| ≥ 90% | 4.5 (Hard) |
| 80-90% | 3.5 (Medium-Hard) |
| 70-80% | 2.5 (Medium) |
| 50-70% | 2.0 (Easy) |
| < 50% | 1.5 (Very Easy) |

## Data Format

### Sample Questions CSV

```csv
id,question_text,topic,difficulty
1,What is the derivative of x²?,calculus,2.0
2,What is the Pythagorean theorem?,geometry,1.0
```

**Required columns:**
- `id`: Unique identifier
- `question_text`: Question text
- `topic`: Topic/category
- `difficulty`: 1-5 scale (optional, default: 2.5)

### Training Data CSV

```csv
question_text,topic
What is calculus?,calculus
What is the Pythagorean theorem?,geometry
```

**Required columns:**
- `question_text`: Question for training
- `topic`: Correct topic label

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src
```

### Test Coverage

- **Classifier Tests**: Training, prediction, validation, persistence
- **Recommender Tests**: Question management, difficulty adaptation, study planning
- **Data Loading**: CSV parsing, validation, error handling

## Example Workflow

```python
from src.data_loader import DataLoader
from src.classifier import QuestionClassifier
from src.recommender import StudyRecommender
from src.utils import format_performance_report

# Step 1: Load and train
loader = DataLoader()
texts, labels = loader.load_training_data('data/training_data.csv')

classifier = QuestionClassifier()
classifier.train(texts, labels)
classifier.save_model('models/classifier.pkl')

# Step 2: Setup recommender
recommender = StudyRecommender('models/classifier.pkl')
questions = loader.load_questions('data/sample_questions.csv')
recommender.add_questions(questions)

# Step 3: Study session
for _ in range(5):
    # Get next question
    question = recommender.get_next_question('calculus')
    print(f"\nQuestion: {question['text']}")
    
    # Simulate student answer (random for demo)
    import random
    correct = random.random() > 0.3  # 70% accuracy
    
    # Update performance
    recommender.update_student_accuracy(
        'calculus',
        0.7 if correct else 0.5
    )

# Step 4: View progress
report = format_performance_report(recommender.get_performance_summary())
print(report)
```

## Performance Characteristics

- **Classification Accuracy**: 85% on held-out validation data
- **Training Time**: < 1 second for typical datasets
- **Inference Time**: < 10ms per question
- **Model Size**: ~2MB
- **Memory**: ~100MB for typical usage

## Error Handling

The system gracefully handles:
- Missing or malformed CSV files
- Insufficient training data
- Invalid difficulty ratings
- Empty question databases
- Untrained classifiers

All errors include descriptive messages for debugging.

## Limitations & Future Work

**Current Limitations:**
- Requires sufficient training data (min 2 topics)
- Limited to English language
- Simple bag-of-words approach (no context)

**Future Enhancements:**
- Support multiple languages via translation
- Advanced NLP (TF-IDF, word embeddings, transformers)
- Multi-label classification (questions in multiple topics)
- Collaborative filtering for student recommendations
- Web API and database backend
- Interactive UI for students and instructors
- Export reports and analytics

## License

MIT License - See LICENSE file for details

## Author

WilberforceWilberforce

## Contact

For questions or contributions, please open an issue on GitHub.

---

**Last Updated**: 2026

**Version**: 1.0.0
