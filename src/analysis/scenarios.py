import copy

def simulate_budget_change(df, model, X, channel, multiplier):
    """
    Simulates spend change for a single channel.
    """
    df_new = copy.deepcopy(df)
    df_new[channel] = df_new[channel] * multiplier

    X_new = X.copy()
    X_new[channel] = X_new[channel] * multiplier

    base_pred = model.predict(X).sum()
    new_pred = model.predict(X_new).sum()

    return {
        "channel": channel,
        "multiplier": multiplier,
        "delta_sales": new_pred - base_pred
    }
