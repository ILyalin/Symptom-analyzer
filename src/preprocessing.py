
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})
    
    return df


df = load_data("..//data//raw//dataset.csv")
df_clean = preprocess_data(df)
df_clean.to_csv("..//data//processed//disease_encoded.csv", index=False)
