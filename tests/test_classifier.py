from network_traffic_ml.data import generate_flow_dataset
from network_traffic_ml.model import evaluate_classifier, train_classifier


def test_flow_dataset_contains_expected_columns():
    data = generate_flow_dataset(n_samples=100, seed=1)
    assert "traffic_class" in data.columns
    assert data["traffic_class"].nunique() >= 3


def test_classifier_training():
    data = generate_flow_dataset(n_samples=300, seed=1)
    model, _, x_test, _, y_test = train_classifier(data, seed=1)
    result = evaluate_classifier(model, x_test, y_test)
    assert 0 <= result.accuracy <= 1
