import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def train_model():
    df = load_processed_data("../data/processed/disease_encoded.csv")

    X = df.drop(columns=['Disease'])
    y = df['Disease']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "..//models//random_forest.pkl")
    joblib.dump(le, "..//models//label_encoder.pkl")

    print("Модель обучена и сохранена!")
train_model()