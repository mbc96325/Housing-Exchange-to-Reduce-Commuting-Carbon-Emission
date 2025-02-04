import pandas as pd
import numpy as np

num_work_days_per_year = 260

emissions_car = {
    'CO2': 252.5,
    'NOx': 0.32,
    'VOC': 0.086,
    'CO': 1.4,
    'SO2': 0.22,
    'PM25': 0.0028,
}

emissions_pt = {
    'CO2': 97.7,
    'NOx': 0.24,
    'VOC': 0.0094,
    'CO': 0.032,
    'SO2': 0.08,
    'PM25': 0.0027,
}

# budget_income_years = 4.0

free_income_prop = {
    'Munich': 1-0.33,
    'Beijing': 1-0.3,
    'Singapore': 1-0.27}

transaction_cost_percent = {
    'Munich': 0.0357,
    'Beijing': 0.02,
    'Singapore': 0.02}


annual_income_growth_rate = {
    'Munich': 0.026,
    'Beijing':  0.081,
    'Singapore': 0.022}

avg_house_increase_per_year = {
    'Munich': np.power(1.5266, 1 / 10) - 1,  # 10 years increase 52.66%
    'Beijing':  np.power(1.7985, 1 / 10) - 1, # 10 years increase 79.85%
    'Singapore': np.power(1 + 1.1668, 1/15) - 1, # 15 years increase 116.68%
}

vot_std = {
    'Munich': 2.2,
    'Beijing':  0.9,
    'Singapore': 0.28}

ev_ratio = {
    'Munich': 0.05,
    'Beijing':  0.10,
    'Singapore': 0.027}