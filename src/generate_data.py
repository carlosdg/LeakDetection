import numpy as np
import pandas as pd


def generate_data(
    initial_liquid=300,
    num_days_train=28,
    num_extra_days_without_leak=14,
    num_days_with_leak=0,
    leak_per_day=0.1*24,
    sensor_noise=0.5,
    mean_liquid_insertion=30,
    mean_liquid_extraction=30,
    std_liquid_insertion=2,
    std_liquid_extraction=2
):
    num_total_days = num_days_train + num_extra_days_without_leak + num_days_with_leak

    insertions = np.random.normal(
        mean_liquid_insertion, std_liquid_insertion, size=num_total_days)
    extractions = np.random.normal(
        mean_liquid_extraction, std_liquid_extraction, size=num_total_days)
    end_day_measures = np.zeros(num_total_days)
    beginning_day_measures = np.zeros(num_total_days+1)
    beginning_day_measures[0] = initial_liquid

    def end_day_calculation(i, leak=False):
        value = (beginning_day_measures[i]
                 + insertions[i]
                 - extractions[i]
                 + np.random.normal(0, sensor_noise))

        if leak:
            value -= leak_per_day

        return value

    for i in range(num_days_train + num_extra_days_without_leak):
        end_day_measures[i] = end_day_calculation(i)
        beginning_day_measures[i+1] = end_day_measures[i]

    for i in range(num_days_train + num_extra_days_without_leak, num_total_days):
        end_day_measures[i] = end_day_calculation(i, True)
        beginning_day_measures[i+1] = end_day_measures[i]

    return pd.DataFrame({
        'insertions': insertions,
        'extractions': extractions,
        'end_day_measures': end_day_measures,
        'beginning_day_measures': beginning_day_measures[:-1],
    })
