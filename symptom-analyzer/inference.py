import pandas as pd
import joblib

def load_model():
    model = joblib.load("..//models//random_forest.pkl")
    le = joblib.load("..//models//label_encoder.pkl")
    return model, le

def predict_disease(features):
    model, le = load_model()
    prediction = model.predict([features])
    return le.inverse_transform(prediction)[0]


new_patient = [1, 0, 1, 1, 20, 0, 1, 1, 1]  
disease = predict_disease(new_patient)
print(f"Предсказанная болезнь: {disease}")