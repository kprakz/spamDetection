from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#Load model and vectorizer
model = joblib.load(r'models//logreg_bow.pkl')
vectorizer = joblib.load(r'vectorizer//bow_vectorizer.pkl')

app = FastAPI()

class Email(BaseModel):
    subject: str

@app.post("/predict")

def predict(email: Email):
    X = vectorizer.transform([email.subject])
    prediction = model.predict(X)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"subject": email.subject, "label": label}