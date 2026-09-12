import matplotlib.pyplot as plt

def plot_actual_vs_pred(df, y_col, pred_col, date_col):
    plt.figure(figsize=(14,5))
    plt.plot(df[date_col], df[y_col], label="Actual")
    plt.plot(df[date_col], df[pred_col], label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.show()
