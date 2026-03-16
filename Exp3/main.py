from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import os

# Create FastAPI app
app = FastAPI()

# Template path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Prediction API
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    income: float = Form(...),
    loan: float = Form(...),
    credit: float = Form(...)
):

    data = np.array([[income, loan, credit]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )