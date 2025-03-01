from sklearn.metrics import mean_absolute_error
def calculate_overall_mae(df, actual_col, pred_cols):
    results = {}
    for pred_col in pred_cols:
        if pred_col in df.columns:
            results[pred_col] = mean_absolute_error(df[actual_col], df[pred_col])
    return results