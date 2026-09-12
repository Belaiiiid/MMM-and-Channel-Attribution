import pandas as pd

def compute_contributions(model, X, channel_cols):
    """
    Decomposes prediction into channel contributions.
    """
    baseline_X = X.copy()
    baseline_X[channel_cols] = 0.0

    baseline_pred = model.predict(baseline_X)
    total_pred = model.predict(X)

    contributions = {}

    for ch in channel_cols:
        X_wo = X.copy()
        X_wo[ch] = 0.0
        contrib = total_pred - model.predict(X_wo)
        contributions[ch] = contrib

    contrib_df = pd.DataFrame(contributions)
    contrib_df["baseline"] = baseline_pred
    contrib_df["total"] = total_pred

    return contrib_df
