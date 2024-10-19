from multiversum import generate_multiverse_grid


def test_grid():
    assert generate_multiverse_grid({
        'x': [1, 2],
        'y': [3, 4]
    }) == [
        {'x': 1, 'y': 3},
        {'x': 1, 'y': 4},
        {'x': 2, 'y': 3},
        {'x': 2, 'y': 4}
    ]
