def train_predict_leak(data, num_days_train):
    observed_variance = (data.end_day_measures
                         - data.insertions
                         + data.extractions
                         - data.beginning_day_measures)
    train_mean = observed_variance[:num_days_train].mean()
    train_std = observed_variance[:num_days_train].std()

    preds = observed_variance[observed_variance < (train_mean - train_std*5)]

    if len(preds) == 0:
        # No leak found
        return -1
    else:
        # Leak found
        return preds.index[0]
