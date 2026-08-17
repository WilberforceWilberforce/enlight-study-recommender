"""
Utility Functions
Helper functions for the study recommender system
"""

import json
from typing import Any, Dict, List


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage."""
    if total == 0:
        return 0.0
    return min(1.0, correct / total)


def get_difficulty_label(difficulty: float) -> str:
    """Convert difficulty score to label."""
    if difficulty < 1.5:
        return "Very Easy"
    elif difficulty < 2.5:
        return "Easy"
    elif difficulty < 3.5:
        return "Medium"
    elif difficulty < 4.5:
        return "Hard"
    else:
        return "Very Hard"


def format_performance_report(performance: Dict[str, Dict]) -> str:
    """Format performance data as readable report."""
    report = []
    report.append("=" * 60)
    report.append("PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")

    for topic, metrics in sorted(performance.items()):
        report.append(f"Topic: {topic}")
        report.append("-" * 60)

        if metrics['accuracy'] is not None:
            accuracy_pct = metrics['accuracy'] * 100
            report.append(f"  Accuracy: {accuracy_pct:.1f}%")
        else:
            report.append("  Accuracy: No data yet")

        if metrics['recommended_difficulty'] is not None:
            diff_label = get_difficulty_label(metrics['recommended_difficulty'])
            report.append(f"  Recommended Level: {diff_label} ({metrics['recommended_difficulty']:.1f}/5)")

        report.append(f"  Available Questions: {metrics['num_questions_available']}")
        report.append("")

    report.append("=" * 60)
    return "\n".join(report)


def merge_predictions(predictions: List[Dict[str, float]],
                     method: str = 'average') -> Dict[str, float]:
    """
    Merge multiple probability predictions.

    Args:
        predictions: List of probability dictionaries
        method: 'average', 'max', or 'min'

    Returns:
        Merged probability dictionary
    """
    if not predictions:
        return {}

    all_topics = set()
    for pred in predictions:
        all_topics.update(pred.keys())

    merged = {}

    for topic in all_topics:
        values = [pred.get(topic, 0) for pred in predictions]

        if method == 'average':
            merged[topic] = sum(values) / len(values)
        elif method == 'max':
            merged[topic] = max(values)
        elif method == 'min':
            merged[topic] = min(values)
        else:
            raise ValueError(f"Unknown merge method: {method}")

    return merged
