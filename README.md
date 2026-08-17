# Enlight Study Question Recommender

An offline study assistant that recommends personalized study questions using NLP and machine learning.

## Overview

Enlight uses Naïve Bayes text classification to categorize study questions by topic and adapts question difficulty based on student accuracy. The system processes and vectorizes words using bag-of-words features with scikit-learn, achieving 85% classification accuracy on validation data.

## Features

- **Topic Classification**: Naïve Bayes classifier for automatic question categorization
- **Adaptive Difficulty**: Adjusts question difficulty based on student performance
- **Bag-of-Words Vectorization**: Efficient text processing with scikit-learn
- **High Accuracy**: 85% classification accuracy on held-out validation data
- **Offline Operation**: No internet connection required

## Tech Stack

- Python 3.8+
- NLP (Natural Language Processing)
- Machine Learning (scikit-learn)
- Pandas

## Project Structure

```
enlight-study-recommender/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── questions.csv
│   └── training_data.csv
├── src/
│   ├── classifier.py
│   ├── recommender.py
│   └── utils.py
├── models/
│   └── naive_bayes_model.pkl
├── tests/
│   └── test_classifier.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/WilberforceWilberforce/enlight-study-recommender.git
cd enlight-study-recommender
pip install -r requirements.txt
```

### Usage

```python
from src.recommender import StudyRecommender

recommender = StudyRecommender()
questions = recommender.get_questions(topic="algebra", difficulty="medium")
```

### Training the Model

```bash
python src/classifier.py --train --data data/training_data.csv
```

### Testing

```bash
pytest tests/
```

## Model Performance

- **Accuracy**: 85% on validation data
- **Algorithm**: Naïve Bayes Classifier
- **Features**: Bag-of-Words vectorization
- **Training Data**: Historical study questions and topics

## License

MIT License - See LICENSE file for details

## Author

[Your Name]
