import pytest
from multiversum import generate_multiverse_grid, apply_constraints

def test_apply_constraints():
    dimensions = {
        "scaler": ["StandardScaler", "MinMaxScaler", "no-scaler"],
        "feature_selector": ["SelectKBest_5", "SelectKBest_10", "use-all-features"]
    }

    constraints = {
        "scaler": [
            {"value": "no-scaler", "allowed_if": {"feature_selector": "use-all-features"}},
            {"value": "MinMaxScaler", "forbidden_if": {"feature_selector": "use-all-features"}}
        ]
    }

    multiverse_grid = generate_multiverse_grid(dimensions)
    filtered_grid = apply_constraints(multiverse_grid, constraints)

    expected_grid = [
        {"scaler": "StandardScaler", "feature_selector": "SelectKBest_5"},
        {"scaler": "StandardScaler", "feature_selector": "SelectKBest_10"},
        {"scaler": "StandardScaler", "feature_selector": "use-all-features"},
        {"scaler": "MinMaxScaler", "feature_selector": "SelectKBest_5"},
        {"scaler": "MinMaxScaler", "feature_selector": "SelectKBest_10"},
        {"scaler": "no-scaler", "feature_selector": "use-all-features"}
    ]

    assert filtered_grid == expected_grid

if __name__ == "__main__":
    pytest.main()
