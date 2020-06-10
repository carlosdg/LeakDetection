def train_predict_leak(data, num_days_train):
    diffs = (data.end_day_measures
             - data.insertions
             + data.extractions
             - data.beginning_day_measures)
    train_mean = diffs[:num_days_train].mean()
    train_std = diffs[:num_days_train].std()

    preds = diffs[diffs < (train_mean - train_std*5)]

    if len(preds) == 0:
        # No leak found
        return -1
    else:
        # Leak found
        return preds.index[0]
