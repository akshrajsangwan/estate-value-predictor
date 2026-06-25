from custom_transformers import *
import joblib
model = joblib.load("EstateValuePredictor.pkl")
print("Model loaded successfully!")
print(type(model))