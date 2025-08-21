
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import re

from sklearn.preprocessing import OneHotEncoder


# read data
asthma_df = pd.read_csv("../data/asthma_disease_data.csv")


def to_snake_case(name):
    """Add underscore before uppercase letters (except the first), then lowercase

    Args:
        name (str): old column name

    Returns:
        str: new column name
    """    # 
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

asthma_df.columns = [to_snake_case(c) for c in asthma_df.columns]

asthma_df = asthma_df.drop(columns = "doctor_in_charge")

## Visualization Dataframe
asthma_vis = asthma_df.copy()
asthma_vis["gender"] = asthma_vis["gender"].map({0: "male", 1: "female"}).astype("category")

binary_cols = [
    'smoking', 'pet_allergy', 'family_history_asthma', 'history_of_allergies', 'eczema', 'hay_fever',
    'gastroesophageal_reflux', 'wheezing', 'shortness_of_breath', 'chest_tightness', 'coughing',
    'nighttime_symptoms', 'exercise_induced'
]

asthma_vis[binary_cols] = asthma_vis[binary_cols].apply(lambda x: x.map({0: 'no', 1: 'yes'}).astype('category'))

asthma_vis['ethnicity'] = asthma_vis['ethnicity'].map({
    0: 'caucasian',
    1: 'african american',
    2: 'asian',
    3: 'other'
}).astype('category')

asthma_vis['education_level'] = asthma_vis['education_level'].map({
    0: 'none',
    1: 'high school',
    2: 'bachelors',
    3: 'higher'
}).astype('category')

asthma_vis["patient_id"] = asthma_vis["patient_id"].astype('object')

## Analysis Dataframe
asthma_df['ethnicity'] = asthma_df['ethnicity'].map({
    0: 'caucasian',
    1: 'african american',
    2: 'asian',
    3: 'other'
}).astype('category')

asthma_df['education_level'] = asthma_df['education_level'].map({
    0: 'none',
    1: 'high school',
    2: 'bachelors',
    3: 'higher'
}).astype('category')

asthma_df = pd.get_dummies(
    asthma_df, columns=['ethnicity', 'education_level'], 
    drop_first = True, 
    dtype = int
)

asthma_df.to_csv("../data/asthma_disease_data_analysis.csv", index=None)