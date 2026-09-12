from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def fit_ridge_mmm(X, y, alpha=1.0):
    """
    Fits Ridge regression MMM.

    Returns:
    - trained model pipeline
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    model.fit(X, y)
    return model
