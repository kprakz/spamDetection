# Spam Detection API
A simple NLP-based spam classifier that detects whether a message is Spam or Not Spam using a trained machine learning model. The API is built with FastAPI and deployed using Railway.

### 1. Features
Text preprocessing (lowercasing, punctuation removal, etc.)

Bow, TF-IDF vectorization

Naïve Bayes/ Logistic Regression classifier

FastAPI backend for predictions

Deployed online for real-time access

### 2. Model
Algorithm: Logistic Regression - 98% accuracy 

Vectorizer: BoW

Dataset: SMS Spam Collection Dataset

### 3. Live Demo
🌐 Hosted on: https://spamdetection-production.up.railway.app

Try sending a POST request to /predict:

IN bash terminal- 
curl -X POST https://spamdetection-production.up.railway.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won a free iPhone. Click here to claim."}'
     
### 4. How to Run Locally
- Clone the repo:
git clone https://github.com/yourusername/spamdetection.git
cd spamdetection

- Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

Run the API:
uvicorn app:app --reload

### 5. Project Structure
spamdetection/
│
├── app.py                          # FastAPI app
├── models/logreg_bow.pkl           # Trained Logistic Regression model
├── vectorizer/bow_vectorizer.pkl   # TF-IDF vectorizer
├── test/response_test.py           # Pre-deifned test phrase
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

### 6. Gmail Automation (Optional)
You can integrate this with the Gmail API to automatically detect spam from your inbox and log the results. (Coming soon...)

### 7. License
MIT License
