import pandas as pd

def compute_roi(df, contrib_df, spend_cols):
    """
    Computes average ROI per channel.
    """
    rows = []

    for ch in spend_cols:
        contrib_col = f"{ch}_sat"   # 🔑 key fix

        spend = df[ch].sum()
        incr = contrib_df[contrib_col].sum()

        roi = incr / spend if spend > 0 else 0

        rows.append({
            "channel": ch.replace("sp_", ""),
            "total_spend": spend,
            "incremental_sales": incr,
            "roi": roi
        })

    return pd.DataFrame(rows).sort_values("roi", ascending=False)

