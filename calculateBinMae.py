from sklearn.metrics import mean_absolute_error
import numpy as np

def calculate_mae_for_bins(df, bins, actual_col, pred_cols):
    results = {}
    for lower, upper in bins:
        mask = (df[actual_col] >= lower) & (df[actual_col] < upper)
        filtered_df = df[mask]
        bin_label = f"{lower}-{upper if upper != np.inf else 'max'}"
        results[bin_label] = {}
        for pred_col in pred_cols:
            if pred_col in filtered_df.columns:
                results[bin_label][pred_col] = mean_absolute_error(filtered_df[actual_col], filtered_df[pred_col])
    return results



