from src.pipeline import load_data, split_data


def test_split_shapes():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
