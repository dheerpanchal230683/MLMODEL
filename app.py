from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['message']
    vec = vectorizer.transform([data])
    prediction = model.predict(vec)

    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)